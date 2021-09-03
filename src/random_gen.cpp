#include <memory>

#include <gsl/gsl_rng.h>

#include "distribution/Categorical.h"
#include "utils/Multithreading.h"

using namespace std;
using namespace tomcat::model;

int main(int argc, char* argv[]) {
    shared_ptr<gsl_rng> random_generator1(gsl_rng_alloc(gsl_rng_mt19937));

    Eigen::VectorXd p(3);
    p << 0.2, 0.3, 0.5;
    Categorical cat(p);

    cout << "Samples using a single random generator." << endl;
    Eigen::VectorXi samples(20);
    for (int i = 0; i < 20; i++) {
        samples(i) = cat.sample(random_generator1, 0)(0, 0);
    }
    cout << samples.transpose() << endl;

    shared_ptr<gsl_rng> random_generator_base(gsl_rng_alloc(gsl_rng_mt19937));
    auto random_generators = split_random_generator(random_generator_base, 2);
    cout << "Samples using a multiple random generators." << endl;
    for (int i = 0; i < 10; i++) {
        samples(i) = cat.sample(random_generator1, 0)(0, 0);
    }
    for (int i = 0; i < 10; i++) {
        samples(i) = cat.sample(random_generator1, 0)(0, 0);
    }
    cout << samples.transpose() << endl;

    shared_ptr<gsl_rng> random_generator_mac(gsl_rng_alloc(gsl_rng_mt19937));
    Eigen::VectorXd pmac(10);
    pmac << 0.11019378383213408734, 0.11019378383213408734,
        0.11019378383213408734, 1.151238009480141500e-08,
        0.11844971783054725201, 0.11019378383213408734, 0.11019378383213408734,
        0.11019378383213408734, 0.11019378383213408734, 0.11019378383213408734;
    Categorical catmac(pmac);
    cout << "Samples from a quasi-uniform distribution (local)." << endl;
    Eigen::VectorXd samples2(10);
    samples2 = catmac.sample_many({random_generator_mac}, 10, 0).col(0);
    cout << samples2.transpose() << endl;

    shared_ptr<gsl_rng> random_generator_server(gsl_rng_alloc(gsl_rng_mt19937));
    Eigen::VectorXd pserver(10);
    pserver << 0.11019378383213405959, 0.11019378383213405959,
        0.11019378383213405959, 1.1512380094801422488e-08,
        0.1184497178305473214, 0.11019378383213405959, 0.11019378383213405959,
        0.11019378383213405959, 0.11019378383213405959, 0.11019378383213405959;
    Categorical catserver(pserver);
    cout << "Samples from a quasi-uniform distribution (server)." << endl;
    samples2 = catserver.sample_many({random_generator_server}, 10, 0).col(0);
    cout << samples2.transpose() << endl;
}