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
        samples(i) = cat.sample(random_generator1, 0)(0,0);
    }
    cout << samples.transpose() << endl;


    shared_ptr<gsl_rng> random_generator_base(gsl_rng_alloc(gsl_rng_mt19937));
    auto random_generators = split_random_generator(random_generator_base, 2);
    cout << "Samples using a multiple random generators." << endl;
    for (int i = 0; i < 10; i++) {
        samples(i) = cat.sample(random_generator1, 0)(0,0);
    }
    for (int i = 0; i < 10; i++) {
        samples(i) = cat.sample(random_generator1, 0)(0,0);
    }
    cout << samples.transpose() << endl;


    shared_ptr<gsl_rng> random_generator2(gsl_rng_alloc(gsl_rng_mt19937));
    Eigen::VectorXd p2(10);
    p2 << 0.110194, 0.110194, 0.110194, 1.15124e-08, 0.11845, 0.110194, 0.110194, 0.110194, 0.110194, 0.110194;
    Categorical cat2(p2);
    cout << "Samples from a quasi-uniform distribution." << endl;
    for (int i = 0; i < 20; i++) {
        samples(i) = cat.sample(random_generator2, 0)(0,0);
    }
    cout << samples.transpose() << endl;
}