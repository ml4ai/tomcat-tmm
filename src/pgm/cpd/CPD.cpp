#include "CPD.h"

#include <algorithm>
#include <thread>

#include "pgm/RandomVariableNode.h"
#include "pgm/TimerNode.h"
#include "utils/EigenExtensions.h"
#include "utils/Multithreading.h"

namespace tomcat {
    namespace model {

        using namespace std;

        //----------------------------------------------------------------------
        // Constructors & Destructor
        //----------------------------------------------------------------------
        CPD::CPD() {}

        CPD::CPD(const vector<shared_ptr<NodeMetadata>>& parent_node_order)
            : parent_node_order(parent_node_order) {
            this->init_id();
            this->fill_indexing_mapping();
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        CPD::CPD(vector<shared_ptr<NodeMetadata>>&& parent_node_order)
            : parent_node_order(move(parent_node_order)) {
            this->init_id();
            this->fill_indexing_mapping();
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        CPD::CPD(const vector<shared_ptr<NodeMetadata>>& parent_node_order,
                 const vector<shared_ptr<Distribution>>& distributions)
            : parent_node_order(parent_node_order),
              distributions(distributions) {
            this->init_id();
            this->fill_indexing_mapping();
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        CPD::CPD(vector<shared_ptr<NodeMetadata>>&& parent_node_order,
                 vector<shared_ptr<Distribution>>&& distributions)
            : parent_node_order(move(parent_node_order)),
              distributions(move(distributions)) {
            this->init_id();
            this->fill_indexing_mapping();
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        CPD::~CPD() {}

        //----------------------------------------------------------------------
        // Operator overload
        //----------------------------------------------------------------------
        ostream& operator<<(ostream& os, const CPD& cpd) {
            cpd.print(os);
            return os;
        }

        //----------------------------------------------------------------------
        // Member functions
        //----------------------------------------------------------------------
        void CPD::init_id() {
            vector<string> labels;
            labels.reserve(this->parent_node_order.size());

            for (const auto& metadata : this->parent_node_order) {
                labels.push_back(metadata->get_label());
            }

            sort(labels.begin(), labels.end());
            stringstream ss;
            copy(labels.begin(),
                 labels.end(),
                 ostream_iterator<string>(ss, ","));
            this->id = ss.str();
        }

        void CPD::fill_indexing_mapping() {
            int cum_cardinality = 1;

            for (int order = this->parent_node_order.size() - 1; order >= 0;
                 order--) {
                shared_ptr<NodeMetadata> metadata =
                    this->parent_node_order[order];

                ParentIndexing indexing(
                    order, metadata->get_cardinality(), cum_cardinality);
                this->parent_label_to_indexing[metadata->get_label()] =
                    indexing;

                cum_cardinality *= metadata->get_cardinality();
            }
        }

        void CPD::copy_cpd(const CPD& cpd) {
            this->id = cpd.id;
            this->parent_label_to_indexing = cpd.parent_label_to_indexing;
            this->parent_node_order = cpd.parent_node_order;
            this->distributions = cpd.distributions;
            this->sufficient_statistics_mutex = make_unique<mutex>();
        }

        void
        CPD::update_dependencies(const Node::NodeMap& parameter_nodes_map) {
            for (auto& distribution : this->distributions) {
                distribution->update_dependencies(parameter_nodes_map);
            }
        }

        Eigen::MatrixXd
        CPD::sample(const vector<shared_ptr<gsl_rng>>& random_generator_per_job,
                    int num_samples,
                    const std::shared_ptr<const RandomVariableNode>& cpd_owner,
                    const std::vector<int>& time_steps_per_sample) const {

            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(cpd_owner->get_parents(),
                                                       num_samples);

            // For rows in which the node time step is bigger than it's supposed
            // to be, we do not sample.
            if (!time_steps_per_sample.empty()) {
                int node_time_step = cpd_owner->get_time_step();
                for (int i = 0; i < distribution_indices.size(); i++) {
                    if (i >= time_steps_per_sample.size())
                        break;

                    int max_time = time_steps_per_sample[i] - 1;
                    if (node_time_step > max_time) {
                        distribution_indices[i] = NO_OBS;
                    }
                }
            }

            int num_jobs = random_generator_per_job.size();
            int sample_size = this->distributions[0]->get_sample_size();
            Eigen::MatrixXd samples(distribution_indices.size(), sample_size);
            mutex samples_mutex;

            const auto processing_blocks =
                get_parallel_processing_blocks(num_jobs, samples.rows());

            if (processing_blocks.size() == 1) {
                // Run in the main thread
                this->run_samples_thread(cpd_owner,
                                         distribution_indices,
                                         random_generator_per_job.at(0),
                                         make_pair(0, samples.rows()),
                                         samples,
                                         samples_mutex);
            }
            else {
                vector<thread> threads;
                for (int i = 0; i < processing_blocks.size(); i++) {
                    thread samples_thread(&CPD::run_samples_thread,
                                          this,
                                          cpd_owner,
                                          ref(distribution_indices),
                                          ref(random_generator_per_job.at(i)),
                                          ref(processing_blocks.at(i)),
                                          ref(samples),
                                          ref(samples_mutex));
                    threads.push_back(move(samples_thread));
                }

                for (auto& samples_thread : threads) {
                    samples_thread.join();
                }
            }

            return samples;
        }

        void CPD::run_samples_thread(
            const std::shared_ptr<const RandomVariableNode>& cpd_owner,
            const Eigen::VectorXi& distribution_indices,
            const shared_ptr<gsl_rng>& random_generator,
            const std::pair<int, int>& processing_block,
            Eigen::MatrixXd& full_samples,
            std::mutex& samples_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            const auto& timer = cpd_owner->get_timer();
            const auto& previous = cpd_owner->get_previous();
            Eigen::MatrixXd samples(num_rows, full_samples.cols());
            for (int i = initial_row; i < initial_row + num_rows; i++) {
                int distribution_idx = distribution_indices[i];
                Eigen::VectorXd sample;

                if (distribution_idx == NO_OBS) {
                    sample = Eigen::VectorXd::Constant(
                        cpd_owner->get_metadata()->get_sample_size(), NO_OBS);
                }
                else {
                    if (cpd_owner->get_metadata()->is_timer()) {
                        bool decrement_sample = false;

                        if (previous) {
                            if (previous->get_assignment()(i, 0) > 0) {
                                decrement_sample = true;
                            }
                        }

                        if (decrement_sample) {
                            sample =
                                previous->get_assignment().row(i).array() - 1;
                        }
                        else {
                            sample =
                                this->distributions[distribution_idx]->sample(
                                    random_generator, i);
                        }
                    }
                    else {
                        bool repeat_sample = false;

                        if (cpd_owner->get_timer()) {
                            const auto& previous_timer =
                                dynamic_pointer_cast<RandomVariableNode>(
                                    cpd_owner->get_timer())
                                    ->get_previous();
                            if (previous_timer) {
                                if (previous_timer->get_assignment()(i, 0) >
                                    0) {
                                    // Node is controlled by a timer and
                                    // previous count value did not reach 0 yet.
                                    repeat_sample = true;
                                }
                            }
                        }

                        if (repeat_sample) {
                            sample = previous->get_assignment().row(i);
                        }
                        else {
                            sample =
                                this->distributions[distribution_idx]->sample(
                                    random_generator, i);
                        }
                    }
                }

                samples.row(i - initial_row) = move(sample);
            }

            scoped_lock lock(samples_mutex);
            full_samples.block(initial_row, 0, num_rows, full_samples.cols()) =
                samples;
        }

        Eigen::MatrixXd CPD::sample_from_posterior(
            const vector<shared_ptr<gsl_rng>>& random_generator_per_job,
            const Eigen::MatrixXd& posterior_weights,
            const std::shared_ptr<const RandomVariableNode>& cpd_owner,
            const std::vector<int>& time_steps_per_sample) const {

            int num_indices = posterior_weights.rows();
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(cpd_owner->get_parents(),
                                                       num_indices);

            // For rows in which the node time step is bigger than it's supposed
            // to be, we do not sample.
            if (!time_steps_per_sample.empty()) {
                int node_time_step = cpd_owner->get_time_step();
                for (int i = 0; i < distribution_indices.size(); i++) {
                    if (i >= time_steps_per_sample.size())
                        break;

                    int max_time = time_steps_per_sample[i] - 1;
                    if (node_time_step > max_time) {
                        distribution_indices[i] = NO_OBS;
                    }
                }
            }

            int num_jobs = random_generator_per_job.size();
            int sample_size = this->distributions[0]->get_sample_size();
            Eigen::MatrixXd samples(distribution_indices.size(), sample_size);
            mutex samples_mutex;

            const auto processing_blocks =
                get_parallel_processing_blocks(num_jobs, samples.rows());

            if (processing_blocks.size() == 1) {
                // Run in the main thread
                this->run_samples_from_posterior_thread(
                    cpd_owner,
                    posterior_weights,
                    distribution_indices,
                    random_generator_per_job.at(0),
                    make_pair(0, samples.rows()),
                    samples,
                    samples_mutex);
            }
            else {
                vector<thread> threads;
                for (int i = 0; i < processing_blocks.size(); i++) {
                    thread samples_thread(
                        &CPD::run_samples_from_posterior_thread,
                        this,
                        cpd_owner,
                        posterior_weights,
                        ref(distribution_indices),
                        ref(random_generator_per_job.at(i)),
                        ref(processing_blocks.at(i)),
                        ref(samples),
                        ref(samples_mutex));
                    threads.push_back(move(samples_thread));
                }

                for (auto& samples_thread : threads) {
                    samples_thread.join();
                }
            }

            return samples;
        }

        void CPD::run_samples_from_posterior_thread(
            const shared_ptr<const RandomVariableNode>& cpd_owner,
            const Eigen::MatrixXd& posterior_weights,
            const Eigen::VectorXi& distribution_indices,
            const shared_ptr<gsl_rng>& random_generator,
            const pair<int, int>& processing_block,
            Eigen::MatrixXd& full_samples,
            mutex& samples_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            const auto& previous = cpd_owner->get_previous();

            Eigen::MatrixXd samples(num_rows, full_samples.cols());
            for (int i = initial_row; i < initial_row + num_rows; i++) {
                int distribution_idx = distribution_indices[i];

                if (distribution_idx == NO_OBS) {
                    samples.row(i - initial_row) = Eigen::VectorXd::Constant(
                        cpd_owner->get_metadata()->get_sample_size(), NO_OBS);
                }
                else {
                    const auto& distribution =
                        this->distributions[distribution_idx];
                    const auto& weights = posterior_weights.row(i);

                    if (cpd_owner->has_timer() && previous) {
                        // If the node that owns this CPD is controlled by a
                        // timer, P(node|Parents(node)) is calculated according
                        // to its left and right segments. If node(t-1) ==
                        // node(t), we must ignore the transition probability
                        // from the transition matrix as there's no transition
                        // to a different state. This would affect only the
                        // duration of the previous segment (or the full segment
                        // if node(t+1)
                        // == node(t) as well) which will be increased by one.
                        // To avoid returning 0 (or whatever value defined in
                        // the transition matrix when node(t) == node(t-1)) we
                        // pass the previous state to the sampling method so
                        // that P(node (t)|node (t-1)) s.t. node(t-1) == node(t)
                        // equals 1. The actual probability of this
                        // "transition", regarding the duration of the left and
                        // right segments, is embedded in the weights passed to
                        // the sampling method.
                        double previous_value =
                            previous->get_assignment()(i, 0);
                        samples.row(i - initial_row) = distribution->sample(
                            random_generator, weights, previous_value);
                    }
                    else {
                        samples.row(i - initial_row) =
                            distribution->sample(random_generator, weights);
                    }
                }
            }

            scoped_lock lock(samples_mutex);
            full_samples.block(initial_row, 0, num_rows, full_samples.cols()) =
                samples;
        }

        Eigen::VectorXi CPD::get_indexed_distribution_indices(
            const vector<shared_ptr<Node>>& index_nodes,
            int num_indices) const {

            Eigen::VectorXi indices = Eigen::VectorXi::Zero(num_indices);
            for (const auto& index_node : index_nodes) {
                const string& label = index_node->get_metadata()->get_label();
                const ParentIndexing& indexing =
                    this->parent_label_to_indexing.at(label);

                if (index_node->get_assignment().rows() > num_indices) {
                    // Ancestral sampling may generate less samples for a
                    // child than the number of samples for its parents in
                    // the case of homogeneous world, for instance. This
                    // only considers samples of the parent nodes (index nodes)
                    // up to a certain row (num_indices).
                    indices = indices.array() +
                              index_node->get_assignment()
                                      .block(0, 0, num_indices, 1)
                                      .col(0)
                                      .cast<int>()
                                      .array() *
                                  indexing.right_cumulative_cardinality;
                }
                else {
                    indices = indices.array() +
                              index_node->get_assignment()
                                      .col(0)
                                      .cast<int>()
                                      .array() *
                                  indexing.right_cumulative_cardinality;
                }
            }

            return indices;
        }

        Eigen::MatrixXd CPD::get_posterior_weights(
            const vector<shared_ptr<Node>>& index_nodes,
            const shared_ptr<RandomVariableNode>& sampled_node,
            const shared_ptr<const RandomVariableNode>& cpd_owner,
            int num_jobs,
            const std::vector<int>& time_steps_per_sample) const {

            if (sampled_node->has_timer() &&
                cpd_owner->get_previous() == sampled_node) {
                // This function is reimplemented in the CategoricalCPD. If it
                // falls here, it's because the state transition CPD does not
                // have a categorical distribution and this is not supported by
                // the current implementation.
                throw TomcatModelException(
                    "This implementation only supports calculating the "
                    "posterior for nodes with categorical distribution, in "
                    "case they are controlled by a timer.");
            }

            int data_size = cpd_owner->get_size();
            int cardinality = sampled_node->get_metadata()->get_cardinality();

            // Set sampled node's assignment equals to zero so we can get the
            // index of the first distribution indexed by this node and the
            // other parent nodes that the child (owner of this CPD) may have.
            Eigen::MatrixXd saved_assignment = sampled_node->get_assignment();
            sampled_node->set_assignment(Eigen::MatrixXd::Zero(data_size, 1));
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(index_nodes, data_size);
            // Restore the sampled node's assignment to its original state.
            sampled_node->set_assignment(saved_assignment);

            // For rows in which the node time step is bigger than it's supposed
            // to be, we do not sample.
            if (!time_steps_per_sample.empty()) {
                int node_time_step = cpd_owner->get_time_step();
                for (int i = 0; i < distribution_indices.size(); i++) {
                    if (i >= time_steps_per_sample.size())
                        break;

                    int max_time = time_steps_per_sample[i] - 1;
                    if (node_time_step > max_time) {
                        distribution_indices[i] = NO_OBS;
                    }
                }
            }

            const string& sampled_node_label =
                sampled_node->get_metadata()->get_label();
            // For every possible value of sampled_node, the offset indicates
            // how many distributions ahead we need to advance to get the
            // distribution indexes by the index nodes of this CPD.
            int distribution_index_offset =
                this->parent_label_to_indexing.at(sampled_node_label)
                    .right_cumulative_cardinality;

            Eigen::MatrixXd weights(data_size, cardinality);
            mutex weights_mutex;

            if (num_jobs == 1) {
                // Run in the main thread
                this->run_posterior_weights_thread(cpd_owner,
                                                   distribution_indices,
                                                   cardinality,
                                                   distribution_index_offset,
                                                   make_pair(0, data_size),
                                                   weights,
                                                   weights_mutex);
            }
            else {
                const vector<pair<int, int>> processing_blocks =
                    get_parallel_processing_blocks(num_jobs, data_size);

                vector<thread> threads;
                for (const auto& processing_block : processing_blocks) {
                    thread weights_thread(&CPD::run_posterior_weights_thread,
                                          this,
                                          cpd_owner,
                                          distribution_indices,
                                          cardinality,
                                          distribution_index_offset,
                                          ref(processing_block),
                                          ref(weights),
                                          ref(weights_mutex));
                    threads.push_back(move(weights_thread));
                }

                for (auto& weights_thread : threads) {
                    weights_thread.join();
                }
            }

            return weights;
        }

        void CPD::run_posterior_weights_thread(
            const shared_ptr<const RandomVariableNode>& cpd_owner,
            const Eigen::VectorXi& distribution_indices,
            int cardinality,
            int distribution_index_offset,
            const pair<int, int>& processing_block,
            Eigen::MatrixXd& full_weights,
            mutex& weights_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(num_rows, cardinality);
            for (int i = initial_row; i < initial_row + num_rows; i++) {
                int distribution_idx = distribution_indices[i];

                if (distribution_idx == NO_OBS ||
                    cpd_owner->get_assignment()(i, 0) == NO_OBS) {
                    weights.row(i - initial_row) =
                        Eigen::VectorXd::Ones(cardinality);
                }
                else {
                    bool use_cdf = false;
                    if (cpd_owner->get_metadata()->is_timer()) {
                        // If the node to which we are computing the weights is
                        // parent of a timer node, we ignore weights in the
                        // middle of a segment and only care for the ones in the
                        // beginning of a new segment.
                        const auto& timer =
                            dynamic_pointer_cast<const TimerNode>(cpd_owner);
                        int timer_value = timer->get_forward_assignment()(i, 0);
                        if (timer_value != 0) {
                            // Middle of a segment
                            continue;
                        }

                        int segment_duration =
                            timer->get_backward_assignment()(i, 0);
                        if (!timer->get_next(segment_duration + 1)) {
                            // If the timer is the last right segment. We use
                            // the CDF instead.
                            use_cdf = true;
                        }
                    }

                    for (int j = 0; j < cardinality; j++) {
                        const auto& distribution =
                            this->distributions[distribution_idx +
                                                j * distribution_index_offset];

                        if (use_cdf) {
                            // p(x >= val)
                            weights(i - initial_row, j) = distribution->get_cdf(
                                cpd_owner->get_assignment()(i, 0) - 1, true);
                        }
                        else {
                            weights(i - initial_row, j) = distribution->get_pdf(
                                cpd_owner->get_assignment().row(i));
                        }
                    }
                }
            }

            scoped_lock lock(weights_mutex);
            full_weights.block(initial_row, 0, num_rows, cardinality) = weights;
        }

        Eigen::MatrixXd CPD::get_particle_timer_posterior_weights(
            const Eigen::VectorXi& distribution_indices,
            const std::shared_ptr<const RandomVariableNode>& sampled_node,
            const std::shared_ptr<const TimerNode>& segment_timer,
            int num_jobs) const {

            int data_size = distribution_indices.size();
            int cardinality = sampled_node->get_metadata()->get_cardinality();

            const string& sampled_node_label =
                sampled_node->get_metadata()->get_label();
            // For every possible value of sampled_node, the offset indicates
            // how many distributions ahead we need to advance to get the
            // distribution indexes by the index nodes of this CPD.
            int distribution_index_offset =
                this->parent_label_to_indexing.at(sampled_node_label)
                    .right_cumulative_cardinality;

            Eigen::MatrixXd weights(data_size, cardinality);
            mutex weights_mutex;

            if (num_jobs == 1) {
                // Run in the main thread
                this->run_particle_timer_posterior_weights_thread(
                    segment_timer,
                    distribution_indices,
                    cardinality,
                    distribution_index_offset,
                    make_pair(0, data_size),
                    weights,
                    weights_mutex);
            }
            else {
                const vector<pair<int, int>> processing_blocks =
                    get_parallel_processing_blocks(num_jobs, data_size);

                vector<thread> threads;
                for (const auto& processing_block : processing_blocks) {
                    thread weights_thread(
                        &CPD::run_particle_timer_posterior_weights_thread,
                        this,
                        segment_timer,
                        distribution_indices,
                        cardinality,
                        distribution_index_offset,
                        ref(processing_block),
                        ref(weights),
                        ref(weights_mutex));
                    threads.push_back(move(weights_thread));
                }

                for (auto& weights_thread : threads) {
                    weights_thread.join();
                }
            }

            return weights;
        }

        void CPD::run_particle_timer_posterior_weights_thread(
            const shared_ptr<const TimerNode>& segment_timer,
            const Eigen::VectorXi& distribution_indices,
            int cardinality,
            int distribution_index_offset,
            const pair<int, int>& processing_block,
            Eigen::MatrixXd& full_weights,
            mutex& weights_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(num_rows, cardinality);
            for (int i = initial_row; i < initial_row + num_rows; i++) {
                int distribution_idx = distribution_indices[i];

                bool use_cdf = false;
                const auto& next_timer =
                    dynamic_pointer_cast<TimerNode>(segment_timer->get_next());
                if (!next_timer ||
                    next_timer->get_forward_assignment()(i, 0) != 0) {
                    use_cdf = true;
                }

                int segment_duration =
                    segment_timer->get_forward_assignment()(i, 0);
                if (next_timer && use_cdf) {
                    // The next timer expands the segment in one more time step.
                    // Since this function is called by methods that use a
                    // template DBN, time steps are processed one at a time and,
                    // therefore, we don't need to look after the next timer as
                    // that time step was not observed yet.
                    segment_duration += 1;
                }

                for (int j = 0; j < cardinality; j++) {
                    const auto& distribution =
                        this->distributions[distribution_idx +
                                            j * distribution_index_offset];

                    if (use_cdf) {
                        // p(x >= val)
                        weights(i - initial_row, j) =
                            distribution->get_cdf(segment_duration - 1, true);
                    }
                    else {
                        weights(i - initial_row, j) =
                            distribution->get_pdf(segment_duration);
                    }
                }
            }

            scoped_lock lock(weights_mutex);
            full_weights.block(initial_row, 0, num_rows, cardinality) = weights;
        }

        Eigen::MatrixXd CPD::get_left_segment_posterior_weights(
            const shared_ptr<TimerNode>& left_segment_timer,
            const Eigen::VectorXi& left_segment_distribution_indices,
            const shared_ptr<RandomVariableNode>& right_segment_state,
            int central_segment_time_step,
            int last_time_step,
            int num_jobs) const {

            int data_size = left_segment_distribution_indices.rows();
            int cardinality = left_segment_timer->get_controlled_node()
                                  ->get_metadata()
                                  ->get_cardinality();
            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(data_size, cardinality);
            mutex weights_mutex;

            Eigen::VectorXi left_segment_values =
                left_segment_timer->get_controlled_node()
                    ->get_assignment()
                    .col(0)
                    .cast<int>();

            // Assumes left segment durations are filled in the forward
            // assignment.
            Eigen::VectorXi left_segment_durations =
                left_segment_timer->get_forward_assignment().col(0).cast<int>();

            // Right segment value does not matter if there's no right
            // segment;
            Eigen::VectorXi right_segment_values;
            Eigen::VectorXi right_segment_durations;
            if (right_segment_state) {
                right_segment_values =
                    right_segment_state->get_assignment().col(0).cast<int>();
                right_segment_durations = right_segment_state->get_timer()
                                              ->get_backward_assignment()
                                              .col(0)
                                              .cast<int>();
            }
            else {
                right_segment_values =
                    Eigen::VectorXi::Constant(data_size, NO_OBS);
                right_segment_durations = Eigen::VectorXi::Zero(data_size);
            }

            if (num_jobs == 1) {
                // Run in the main thread
                this->run_left_segment_posterior_weights_thread(
                    left_segment_timer,
                    left_segment_values,
                    left_segment_durations,
                    right_segment_values,
                    right_segment_durations,
                    central_segment_time_step,
                    last_time_step,
                    left_segment_distribution_indices,
                    make_pair(0, data_size),
                    weights,
                    weights_mutex);
            }
            else {
                vector<thread> threads;
                const vector<pair<int, int>> processing_blocks =
                    get_parallel_processing_blocks(num_jobs, data_size);
                for (const auto& processing_block : processing_blocks) {
                    thread weights_thread(
                        &CPD::run_left_segment_posterior_weights_thread,
                        this,
                        ref(left_segment_timer),
                        left_segment_values,
                        left_segment_durations,
                        right_segment_values,
                        right_segment_durations,
                        central_segment_time_step,
                        last_time_step,
                        left_segment_distribution_indices,
                        ref(processing_block),
                        ref(weights),
                        ref(weights_mutex));
                    threads.push_back(move(weights_thread));
                }

                for (auto& weights_thread : threads) {
                    weights_thread.join();
                }
            }

            return weights;
        }

        Eigen::MatrixXd CPD::get_left_segment_posterior_weights(
            const shared_ptr<TimerNode>& left_segment_timer,
            const shared_ptr<RandomVariableNode>& right_segment_state,
            int central_segment_time_step,
            int last_time_step,
            int num_jobs) const {

            int data_size = left_segment_timer->get_size();
            int cardinality = left_segment_timer->get_controlled_node()
                                  ->get_metadata()
                                  ->get_cardinality();
            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(data_size, cardinality);
            mutex weights_mutex;

            if (num_jobs == 1) {
                this->run_single_left_segment_posterior_weights_thread(
                    left_segment_timer,
                    right_segment_state,
                    central_segment_time_step,
                    last_time_step,
                    make_pair(0, data_size),
                    weights,
                    weights_mutex);
            }
            else {
                vector<thread> threads;
                const vector<pair<int, int>> processing_blocks =
                    get_parallel_processing_blocks(num_jobs, data_size);
                for (const auto& processing_block : processing_blocks) {
                    thread weights_thread(
                        &CPD::run_single_left_segment_posterior_weights_thread,
                        this,
                        ref(left_segment_timer),
                        ref(right_segment_state),
                        central_segment_time_step,
                        last_time_step,
                        ref(processing_block),
                        ref(weights),
                        ref(weights_mutex));
                    threads.push_back(move(weights_thread));
                }

                for (auto& weights_thread : threads) {
                    weights_thread.join();
                }
            }

            return weights;
        }

        void CPD::run_single_left_segment_posterior_weights_thread(
            const shared_ptr<TimerNode>& left_segment_last_timer,
            const shared_ptr<RandomVariableNode>& right_segment_state,
            int central_segment_time_step,
            int last_time_step,
            const pair<int, int>& processing_block,
            Eigen::MatrixXd& full_weights,
            mutex& weights_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;
            int cardinality = full_weights.cols();

            Eigen::MatrixXd weights(num_rows, cardinality);

            for (int i = initial_row; i < initial_row + num_rows; i++) {
                // Get posterior weights for the left segment of a given
                // timer. This is done by calculating the posterior weights of
                // the timer in the beginning of the left segment. As a node can
                // have multiple assignments at a time, a timer may have several
                // left segments at a time (one for each observation series),
                // therefore, the timer at the beginning of the left segment
                // will vary from row to row of the timer's assignment.
                int left_segment_duration =
                    left_segment_last_timer->get_forward_assignment()(i, 0);

                // Timer in the beginning of the left segment for the i-th
                // assignment
                const auto& left_segment_first_timer =
                    dynamic_pointer_cast<TimerNode>(
                        left_segment_last_timer->get_previous(
                            left_segment_duration));
                weights.row(i - initial_row) =
                    left_segment_first_timer
                        ->get_left_segment_posterior_weights(
                            left_segment_duration,
                            right_segment_state,
                            central_segment_time_step,
                            last_time_step,
                            i);
            }

            scoped_lock lock(weights_mutex);
            full_weights.block(initial_row, 0, num_rows, cardinality) = weights;
        }

        void CPD::run_left_segment_posterior_weights_thread(
            const shared_ptr<TimerNode>& left_segment_timer,
            const Eigen::VectorXi& left_segment_values,
            const Eigen::VectorXi& left_segment_durations,
            const Eigen::VectorXi& right_segment_values,
            const Eigen::VectorXi& right_segment_durations,
            int central_segment_time_step,
            int last_time_step,
            const Eigen::VectorXi& distribution_indices,
            const pair<int, int>& processing_block,
            Eigen::MatrixXd& full_weights,
            mutex& weights_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;
            int cardinality = full_weights.cols();

            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(num_rows, cardinality);

            const Eigen::VectorXi& sub_left_segment_values =
                left_segment_values.block(initial_row, 0, num_rows, 1);
            const Eigen::VectorXi& sub_right_segment_values =
                right_segment_values.block(initial_row, 0, num_rows, 1);

            const Eigen::VectorXi& sub_left_segment_durations =
                left_segment_durations.block(initial_row, 0, num_rows, 1);
            const Eigen::VectorXi& sub_right_segment_durations =
                right_segment_durations.block(initial_row, 0, num_rows, 1);

            for (int j = 0; j < cardinality; j++) {
                Eigen::VectorXi left_joins_central =
                    (sub_left_segment_values.array() == j).cast<int>();
                Eigen::VectorXi central_joins_right =
                    (j == sub_right_segment_values.array()).cast<int>();

                Eigen::VectorXi durations =
                    sub_left_segment_durations.array() +
                    left_joins_central.array() *
                        (1 + central_joins_right.array() *
                                 (sub_right_segment_durations.array() + 1));

                for (int i = initial_row; i < initial_row + num_rows; i++) {
                    int distribution_idx = distribution_indices[i];
                    const auto& timer_distribution =
                        this->distributions[distribution_idx];

                    bool use_cdf = false;
                    if (left_joins_central(i - initial_row) == 1) {
                        if (central_joins_right(i - initial_row) == 1) {
                            if (central_segment_time_step +
                                    right_segment_durations[i - initial_row] +
                                    1 >=
                                last_time_step) {
                                // There's more different segments to the right
                                // side.
                                use_cdf = true;
                            }
                        }
                        else {
                            if (central_segment_time_step == last_time_step) {
                                // There's more different segments to the right
                                // side.
                                use_cdf = true;
                            }
                        }
                    }

                    double d = durations[i - initial_row];
                    double weight;
                    if (use_cdf) {
                        // This deal with the fact that the duration is
                        // truncated at the last time step of the unrolled
                        // DBN. p(x >= d);
                        weight = timer_distribution->get_cdf(d - 1, true);
                    }
                    else {
                        weight = timer_distribution->get_pdf(d);
                    }

                    weights(i - initial_row, j) = weight;
                }
            }

            scoped_lock lock(weights_mutex);
            full_weights.block(initial_row, 0, num_rows, cardinality) = weights;
        }

        Eigen::VectorXd CPD::get_single_left_segment_posterior_weights(
            const shared_ptr<const TimerNode>& left_segment_first_timer,
            int left_segment_duration,
            const shared_ptr<RandomVariableNode>& right_segment_state,
            int central_segment_time_step,
            int last_time_step,
            int sample_idx) const {

            int left_segment_value =
                left_segment_first_timer->get_controlled_node()
                    ->get_assignment()(sample_idx, 0);

            // Right segment value does not matter if there's no right segment;
            int right_segment_value = NO_OBS;
            int right_segment_duration = 0;
            if (right_segment_state) {
                right_segment_value =
                    right_segment_state->get_assignment()(sample_idx, 0);
                right_segment_duration =
                    right_segment_state->get_timer()->get_backward_assignment()(
                        sample_idx, 0);
            }

            int distribution_idx = this->get_indexed_distribution_idx(
                left_segment_first_timer->get_parents(), sample_idx);

            int cardinality = left_segment_first_timer->get_controlled_node()
                                  ->get_metadata()
                                  ->get_cardinality();
            Eigen::VectorXd weights(cardinality);
            for (int i = 0; i < cardinality; i++) {
                int left_joins_central = left_segment_value == i;
                int central_joins_right = i == right_segment_value;

                int d =
                    left_segment_duration +
                    left_joins_central * (1 + central_joins_right *
                                                  (right_segment_duration + 1));

                // We get the timer's distribution at the beginning of the
                // segment because timer nodes can have other dependencies.
                // We need to use these dependencies' assignments to
                // correctly retrieve the actual distribution that governs
                // the timer node at its time step.
                const auto& timer_distribution =
                    this->distributions.at(distribution_idx);

                bool use_cdf = false;
                if (left_joins_central == 1) {
                    if (central_joins_right == 1) {
                        if (central_segment_time_step + right_segment_duration +
                                1 >=
                            last_time_step) {
                            // There's more different segments to the right
                            // side.
                            use_cdf = true;
                        }
                    }
                    else {
                        if (central_segment_time_step == last_time_step) {
                            // There's more different segments to the right
                            // side.
                            use_cdf = true;
                        }
                    }
                }

                double weight;
                if (use_cdf) {
                    // This deal with the fact that the duration is truncated
                    // at the last time step of the unrolled DBN.
                    // p(x >= d);
                    weight = timer_distribution->get_cdf(d - 1, true);
                }
                else {
                    weight = timer_distribution->get_pdf(
                        Eigen::VectorXd::Constant(1, d));
                }

                weights(i) = weight;
            }

            return weights;
        }

        Eigen::MatrixXd CPD::get_central_segment_posterior_weights(
            const shared_ptr<RandomVariableNode>& left_segment_state,
            const shared_ptr<TimerNode>& central_segment_timer,
            const shared_ptr<RandomVariableNode>& right_segment_state,
            int last_time_step,
            int num_jobs) const {

            int data_size = central_segment_timer->get_size();
            int cardinality = central_segment_timer->get_controlled_node()
                                  ->get_metadata()
                                  ->get_cardinality();

            // Set sampled node's assignment to zero so we can
            // get the index of the first distribution indexed by the this node
            // given that the other index nodes have fixed assignments.
            const auto& time_controlled_node =
                central_segment_timer->get_controlled_node();
            Eigen::MatrixXd saved_assignment =
                time_controlled_node->get_assignment();
            time_controlled_node->set_assignment(
                Eigen::MatrixXd::Zero(data_size, 1));
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(
                    central_segment_timer->get_parents(), data_size);
            // Restore the controlled node's assignment to its original state.
            time_controlled_node->set_assignment(saved_assignment);

            const string& sampled_node_label =
                time_controlled_node->get_metadata()->get_label();
            // For every possible value of sampled_node, the offset
            // indicates how many distributions ahead we need to advance to
            // get the distribution indexed by the index nodes of this CPD if
            // we only change the value of the controlled node.
            int distribution_index_offset =
                this->parent_label_to_indexing.at(sampled_node_label)
                    .right_cumulative_cardinality;

            // Get values of the immediate left and right segments
            Eigen::VectorXi left_segment_values;
            if (left_segment_state) {
                left_segment_values =
                    left_segment_state->get_assignment().col(0).cast<int>();
            }
            else {
                left_segment_values =
                    Eigen::VectorXi::Constant(data_size, NO_OBS);
            }

            Eigen::VectorXi central_segment_values =
                central_segment_timer->get_controlled_node()
                    ->get_assignment()
                    .col(0)
                    .cast<int>();

            Eigen::VectorXi right_segment_values;
            Eigen::VectorXi right_segment_durations;
            if (right_segment_state) {
                right_segment_values =
                    right_segment_state->get_assignment().col(0).cast<int>();
                right_segment_durations = right_segment_state->get_timer()
                                              ->get_backward_assignment()
                                              .col(0)
                                              .cast<int>();
            }
            else {
                right_segment_values =
                    Eigen::VectorXi::Constant(data_size, NO_OBS);
                right_segment_durations = Eigen::VectorXi::Zero(data_size);
            }

            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(data_size, cardinality);
            mutex weights_mutex;
            if (num_jobs == 1) {
                // Run in the main thread
                this->run_central_segment_posterior_weights_thread(
                    central_segment_timer,
                    left_segment_values,
                    right_segment_values,
                    right_segment_durations,
                    last_time_step,
                    distribution_indices,
                    distribution_index_offset,
                    make_pair(0, data_size),
                    weights,
                    weights_mutex);
            }
            else {
                vector<thread> threads;
                const vector<pair<int, int>> processing_blocks =
                    get_parallel_processing_blocks(num_jobs, data_size);
                for (const auto& processing_block : processing_blocks) {
                    thread weights_thread(
                        &CPD::run_central_segment_posterior_weights_thread,
                        this,
                        ref(central_segment_timer),
                        left_segment_values,
                        right_segment_values,
                        right_segment_durations,
                        last_time_step,
                        distribution_indices,
                        distribution_index_offset,
                        ref(processing_block),
                        ref(weights),
                        ref(weights_mutex));
                    threads.push_back(move(weights_thread));
                }

                for (auto& weights_thread : threads) {
                    weights_thread.join();
                }
            }

            return weights;
        }

        void CPD::run_central_segment_posterior_weights_thread(
            const shared_ptr<TimerNode>& central_segment_timer,
            const Eigen::VectorXi& left_segment_values,
            const Eigen::VectorXi& right_segment_values,
            const Eigen::VectorXi& right_segment_durations,
            int last_time_step,
            const Eigen::VectorXi& distribution_indices,
            int distribution_index_offset,
            const pair<int, int>& processing_block,
            Eigen::MatrixXd& full_weights,
            mutex& weights_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;
            int cardinality = full_weights.cols();

            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(num_rows, cardinality);

            const Eigen::VectorXi& sub_left_segment_values =
                left_segment_values.block(initial_row, 0, num_rows, 1);
            const Eigen::VectorXi& sub_right_segment_values =
                right_segment_values.block(initial_row, 0, num_rows, 1);

            const Eigen::VectorXi& sub_right_segment_durations =
                right_segment_durations.block(initial_row, 0, num_rows, 1);

            for (int j = 0; j < cardinality; j++) {
                Eigen::VectorXi left_does_not_join_central =
                    (sub_left_segment_values.array() != j).cast<int>();
                Eigen::VectorXi central_joins_right =
                    (j == sub_right_segment_values.array()).cast<int>();

                // Logic array (0'' and 1's) to indicate the assignment indices
                // where the central and right segments have equal value.
                Eigen::VectorXi durations =
                    (central_joins_right.array() *
                     (1 + sub_right_segment_durations.array()));

                for (int i = initial_row; i < initial_row + num_rows; i++) {
                    if (left_does_not_join_central(i - initial_row) == 1) {
                        int distribution_idx = distribution_indices[i] +
                                               j * distribution_index_offset;
                        const auto& timer_distribution =
                            this->distributions[distribution_idx];
                        double d =
                            durations.row(i - initial_row).cast<double>()(0);

                        double weight;
                        if (central_segment_timer->get_time_step() + d <
                            last_time_step) {
                            weight = timer_distribution->get_pdf(d);
                        }
                        else {
                            // This deal with the fact that the duration is
                            // truncated at the last time step of the unrolled
                            // DBN. p(x >= d);
                            weight = timer_distribution->get_cdf(d - 1, true);
                        }

                        weights(i - initial_row, j) = weight;
                    }
                }
            }

            scoped_lock lock(weights_mutex);
            full_weights.block(initial_row, 0, num_rows, cardinality) = weights;
        }

        Eigen::MatrixXd CPD::get_right_segment_posterior_weights(
            const shared_ptr<TimerNode>& right_segment_first_timer,
            int last_time_step,
            int num_jobs) const {

            int data_size = right_segment_first_timer->get_size();

            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(
                    right_segment_first_timer->get_parents(), data_size);

            const Eigen::VectorXi& right_segment_values =
                right_segment_first_timer->get_controlled_node()
                    ->get_assignment()
                    .col(0)
                    .cast<int>();
            const Eigen::VectorXi& right_segment_durations =
                right_segment_first_timer->get_backward_assignment()
                    .col(0)
                    .cast<int>();

            int cardinality = right_segment_first_timer->get_controlled_node()
                                  ->get_metadata()
                                  ->get_cardinality();
            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(data_size, cardinality);
            mutex weights_mutex;
            if (num_jobs == 1) {
                // Run in the main thread
                this->run_right_segment_posterior_weights_thread(
                    right_segment_first_timer,
                    right_segment_values,
                    right_segment_durations,
                    last_time_step,
                    distribution_indices,
                    make_pair(0, data_size),
                    weights,
                    weights_mutex);
            }
            else {
                vector<thread> threads;
                const vector<pair<int, int>> processing_blocks =
                    get_parallel_processing_blocks(num_jobs, data_size);
                for (const auto& processing_block : processing_blocks) {
                    thread weights_thread(
                        &CPD::run_right_segment_posterior_weights_thread,
                        this,
                        ref(right_segment_first_timer),
                        right_segment_values,
                        right_segment_durations,
                        last_time_step,
                        ref(distribution_indices),
                        ref(processing_block),
                        ref(weights),
                        ref(weights_mutex));
                    threads.push_back(move(weights_thread));
                }

                for (auto& weights_thread : threads) {
                    weights_thread.join();
                }
            }

            return weights;
        }

        void CPD::run_right_segment_posterior_weights_thread(
            const shared_ptr<TimerNode>& right_segment_first_timer,
            const Eigen::VectorXi& right_segment_values,
            const Eigen::VectorXi& right_segment_durations,
            int last_time_step,
            const Eigen::VectorXi& distribution_indices,
            const std::pair<int, int>& processing_block,
            Eigen::MatrixXd& full_weights,
            mutex& weights_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;
            int cardinality = full_weights.cols();

            Eigen::MatrixXd weights =
                Eigen::MatrixXd::Ones(num_rows, cardinality);

            for (int i = initial_row; i < initial_row + num_rows; i++) {
                int distribution_idx = distribution_indices[i];

                const auto& timer_distribution =
                    this->distributions[distribution_idx];

                double d = right_segment_durations.row(i).cast<double>()(0);

                double p_duration_right;
                if (d + right_segment_first_timer->get_time_step() <
                    last_time_step) {
                    p_duration_right = timer_distribution->get_pdf(d);
                }
                else {
                    // This deal with the fact that the duration is
                    // truncated at the last time step of the unrolled
                    // DBN. p(x >= d);
                    p_duration_right = timer_distribution->get_cdf(d - 1, true);
                }

                weights.row(i - initial_row) =
                    Eigen::VectorXd::Constant(cardinality, p_duration_right);

                // Ignore if value of the controlled node in the right
                // segment is equal to the value of the controlled node in
                // the central segment.
                int j = right_segment_values(i, 0);
                weights(i - initial_row, j) = 1;
            }

            scoped_lock lock(weights_mutex);
            full_weights.block(initial_row, 0, num_rows, cardinality) = weights;
        }

        vector<DistributionPtr> CPD::get_posterior(
            const Eigen::MatrixXd& posterior_weights,
            const std::shared_ptr<const RandomVariableNode>& cpd_owner,
            int num_jobs) const {

            int num_indices = posterior_weights.rows();
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(cpd_owner->get_parents(),
                                                       num_indices);

            int sample_size = cpd_owner->get_metadata()->get_sample_size();
            vector<DistributionPtr> posterior_distributions(num_indices);
            mutex posterior_mutex;

            const auto processing_blocks =
                get_parallel_processing_blocks(num_jobs, num_indices);

            if (processing_blocks.size() == 1) {
                // Run in the main thread
                this->run_get_posterior_thread(posterior_weights,
                                               distribution_indices,
                                               make_pair(0, num_indices),
                                               posterior_distributions,
                                               posterior_mutex);
            }
            else {
                vector<thread> threads;
                for (int i = 0; i < processing_blocks.size(); i++) {
                    thread posterior_thread(&CPD::run_get_posterior_thread,
                                            this,
                                            ref(posterior_weights),
                                            ref(distribution_indices),
                                            ref(processing_blocks.at(i)),
                                            ref(posterior_distributions),
                                            ref(posterior_mutex));
                    threads.push_back(move(posterior_thread));
                }

                for (auto& posterior_thread : threads) {
                    posterior_thread.join();
                }
            }

            return posterior_distributions;
        }

        void CPD::run_get_posterior_thread(
            const Eigen::MatrixXd& posterior_weights,
            const Eigen::VectorXi& distribution_indices,
            const pair<int, int>& processing_block,
            vector<DistributionPtr>& full_distributions,
            mutex& posterior_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            vector<DistributionPtr> posteriors(num_rows);
            for (int i = initial_row; i < initial_row + num_rows; i++) {
                int distribution_idx = distribution_indices[i];
                const auto& distribution =
                    this->distributions[distribution_idx];

                posteriors[i - initial_row] = move(distribution->clone());
                posteriors[i - initial_row]->update_from_posterior(
                    posterior_weights.row(i));
            }

            scoped_lock lock(posterior_mutex);
            copy(posteriors.begin(),
                 posteriors.end(),
                 full_distributions.begin() + initial_row);
        }

        int CPD::get_indexed_distribution_idx(
            const std::vector<std::shared_ptr<Node>>& index_nodes,
            int sample_idx) const {

            int distribution_idx = 0;
            for (const auto& index_node : index_nodes) {
                const string& label = index_node->get_metadata()->get_label();
                const ParentIndexing& indexing =
                    this->parent_label_to_indexing.at(label);

                if (sample_idx >= index_node->get_assignment().rows()) {
                    // Ancestral sampling may generate less samples for a
                    // child than the number of samples for its parents in
                    // the case of homogeneous world, for instance. This
                    // only considers samples of the parent nodes (index nodes)
                    // up to a certain row (num_indices).
                    sample_idx = index_node->get_assignment().rows() - 1;
                }

                distribution_idx +=
                    index_node->get_assignment()(sample_idx, 0) *
                    indexing.right_cumulative_cardinality;
            }

            return distribution_idx;
        }

        void CPD::update_sufficient_statistics(
            const shared_ptr<RandomVariableNode>& cpd_owner) {

            int data_size = cpd_owner->get_size();
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(cpd_owner->get_parents(),
                                                       data_size);
            unordered_map<int, vector<double>> values_per_distribution;

            const Eigen::MatrixXd& values =
                cpd_owner->get_metadata()->is_timer()
                    ? dynamic_pointer_cast<TimerNode>(cpd_owner)
                          ->get_forward_assignment()
                    : cpd_owner->get_assignment();

            for (int i = 0; i < data_size; i++) {
                int distribution_idx = distribution_indices[i];

                // TODO - this needs to be adapted for multivariate variables
                double value = values(i, 0);
                bool add_to_list = false;
                if (value != NO_OBS) {
                    if (cpd_owner->get_metadata()->is_timer()) {
                        // Only add to the sufficient statistics, when the timer
                        // starts over which is the moment when a transition
                        // between different states occur.
                        if (const auto& next_timer =
                                dynamic_pointer_cast<TimerNode>(
                                    cpd_owner->get_next())) {
                            // The next timer starts a new segment
                            if (next_timer->get_forward_assignment()(i, 0) ==
                                0) {
                                add_to_list = true;
                            }
                        }
                    }
                    else {
                        if (cpd_owner->has_timer()) {
                            const auto& previous = cpd_owner->get_previous();
                            if (!previous ||
                                previous->get_assignment()(i, 0) != value) {
                                // If a node is controlled by a timer, only add
                                // to the sufficient statistics if there was a
                                // a state transition or when t = 0 (there's no
                                // previous node in time);
                                add_to_list = true;
                            }
                        }
                        else {
                            add_to_list = true;
                        }
                    }

                    if (add_to_list) {
                        if (!EXISTS(distribution_idx,
                                    values_per_distribution)) {
                            values_per_distribution[distribution_idx] = {};
                        }
                        values_per_distribution[distribution_idx].push_back(
                            value);
                    }
                }
            }

            for (auto& [distribution_idx, values] : values_per_distribution) {
                const shared_ptr<Distribution>& distribution =
                    this->distributions[distribution_idx];
                distribution->update_sufficient_statistics(values);
            }
        }

        void CPD::print(ostream& os) const { os << this->get_description(); }

        string CPD::get_description() const {
            stringstream ss;

            int i;
            int size = this->parent_node_order.size() - 1;
            for (i = 0; i < size; i++) {
                ss << this->parent_node_order.at(i)->get_label() << ", ";
            }
            if (!this->parent_node_order.empty()) {
                ss << this->parent_node_order.at(i)->get_label();
            }
            ss << "\n";
            ss << this->get_table(0) << "\n";
            ss << this->get_name() << "\n";

            return ss.str();
        }

        Eigen::MatrixXd CPD::get_table(int parameter_idx) const {
            Eigen::MatrixXd table;

            if (this->frozen_distributions) {
                table = this->distribution_table;
            }
            else {
                int row = 0;
                for (const auto& distribution : this->distributions) {
                    Eigen::VectorXd parameters =
                        distribution->get_values(parameter_idx);
                    if (table.size() == 0) {
                        table = Eigen::MatrixXd(this->distributions.size(),
                                                parameters.size());
                        table.row(row) = parameters;
                    }
                    else {
                        table.row(row) = parameters;
                    }
                    row++;
                }
            }

            return table;
        }

        shared_ptr<CPD> CPD::create_from_data(const EvidenceSet& data,
                                              const string& cpd_owner_label,
                                              int cpd_owner_cardinality) {
            throw TomcatModelException("Only implemented to categorical "
                                       "CPDs.");
        }

        void CPD::freeze_distributions(int parameter_idx) {
            this->frozen_distributions = false;
            this->distribution_table = this->get_table(parameter_idx);
            this->frozen_distributions = true;
        }

        Eigen::VectorXd
        CPD::get_pdfs(const shared_ptr<const RandomVariableNode>& cpd_owner,
                      int num_jobs,
                      int parameter_idx) const {

            int num_samples = cpd_owner->get_size();
            Eigen::VectorXi distribution_indices =
                this->get_indexed_distribution_indices(cpd_owner->get_parents(),
                                                       num_samples);
            Eigen::VectorXd pdfs(num_samples);
            mutex pdfs_mutex;
            if (num_jobs == 1) {
                // Run in the main thread
                this->run_pdf_thread(cpd_owner,
                                     distribution_indices,
                                     parameter_idx,
                                     pdfs,
                                     make_pair(0, num_samples),
                                     pdfs_mutex);
            }
            else {
                const auto processing_blocks =
                    get_parallel_processing_blocks(num_jobs, num_samples);
                vector<thread> threads;
                for (int i = 0; i < processing_blocks.size(); i++) {
                    thread pdfs_thread(&CPD::run_pdf_thread,
                                       this,
                                       ref(cpd_owner),
                                       ref(distribution_indices),
                                       parameter_idx,
                                       ref(pdfs),
                                       ref(processing_blocks.at(i)),
                                       ref(pdfs_mutex));
                    threads.push_back(move(pdfs_thread));
                }

                for (auto& samples_thread : threads) {
                    samples_thread.join();
                }
            }

            return pdfs;
        }

        void CPD::run_pdf_thread(
            const shared_ptr<const RandomVariableNode>& cpd_owner,
            const Eigen::VectorXi& distribution_indices,
            int parameter_idx,
            Eigen::VectorXd& full_pdfs,
            const pair<int, int>& processing_block,
            mutex& pdf_mutex) const {

            int initial_row = processing_block.first;
            int num_rows = processing_block.second;

            Eigen::VectorXd pdfs(num_rows, 1);
            for (int i = initial_row; i < initial_row + num_rows; i++) {
                pdfs[i] = this->distributions[i]->get_pdf(
                    cpd_owner->get_assignment().row(i));
            }

            scoped_lock lock(pdf_mutex);
            full_pdfs.block(initial_row, 0, num_rows, 1) = pdfs;
        }

        //------------------------------------------------------------------
        // Getters & Setters
        //------------------------------------------------------------------
        const string& CPD::get_id() const { return id; }

        const CPD::TableOrderingMap& CPD::get_parent_label_to_indexing() const {
            return parent_label_to_indexing;
        }

        const vector<std::shared_ptr<Distribution>>&
        CPD::get_distributions() const {
            return distributions;
        }

    } // namespace model
} // namespace tomcat
