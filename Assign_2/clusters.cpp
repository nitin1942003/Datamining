#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <memory>

using namespace std;

// representation of point;
// x,y - coordinates on XY
// cluster - assigned cluster, by default -1(not assigned)
struct point
{
    double x, y;
    int cluster = -1;
};

int main()
{
    vector<point> init; // vector of points
    unsigned int k;     // k - number of clusters
    double epsilon;     // threshold for stopping condition
    string filename;    // name of input file

    cin >> k >> epsilon >> filename; // input
    ifstream file(filename);

    // reading coordinates of points
    while (!file.eof())
    {
        point tmp;
        file >> tmp.x >> tmp.y;
        init.push_back(tmp);
    }
    file.close();

    // Randomly select k unique points from the input as initial centers
    vector<point> centers(k);
    vector<int> chosen_indices;
    while (chosen_indices.size() < k)
    {
        int rand_index = rand() % init.size(); // Choose a random index
        if (find(chosen_indices.begin(), chosen_indices.end(), rand_index) == chosen_indices.end())
        {
            centers[chosen_indices.size()] = init[rand_index]; // Add the point as a center
            chosen_indices.push_back(rand_index);
        }
    }

    vector<point> prev_centers(k); // store previous centers

    int itrations = 0;
    ofstream ofile("out.txt");

    // Initial random points taken 
    ofile << "Random points selected from data as mean of clusters initially" << endl;
    for (unsigned int i = 0; i < k; i++)
    {
        ofile << "Cluster " << i + 1 << " mean: ";
        ofile << "(" << centers[i].x << ", " << centers[i].y <<")";
        ofile << "\n";
    }
    ofile << "\n";

    // loop until the difference between previous and current centers is less than epsilon
    while (true)
    {
        itrations++;

        // calculate distance from point to centers for every point
        for (unsigned int j = 0; j < init.size(); j++)
        {
            double *dists = new double[k];
            for (unsigned int p = 0; p < k; p++)
            {
                double a = init[j].y - centers[p].y;    // length in y-axis
                double b = init[j].x - centers[p].x;    // length in x-axis
                dists[p] = sqrt(pow(a, 2) + pow(b, 2)); // distance from point to center
            }
            // assign cluster with closest center
            init[j].cluster = min_element(dists, dists + k) - dists;
            delete[] dists;
        }

        // calculating new centers
        vector<double> sum_x(k, 0), sum_y(k, 0);
        vector<int> count(k, 0);

        for (unsigned int f = 0; f < init.size(); f++)
        {
            sum_x[init[f].cluster] += init[f].x;
            sum_y[init[f].cluster] += init[f].y;
            count[init[f].cluster]++;
        }

        // set new centers to average coordinate of points in cluster
        for (unsigned int f = 0; f < k; f++)
        {
            if (count[f] > 0)
            {
                centers[f].x = sum_x[f] / count[f];
                centers[f].y = sum_y[f] / count[f];
            }
        }

        // check for convergence: difference between previous and current centers
        bool converged = true;
        for (unsigned int f = 0; f < k; f++)
        {
            double dist = sqrt(pow(centers[f].x - prev_centers[f].x, 2) +
                               pow(centers[f].y - prev_centers[f].y, 2));
            if (dist >= epsilon)
            {
                converged = false;
            }
        }

        vector<vector<point>> clusters(k); // Create k clusters
        for (unsigned int i = 0; i < init.size(); i++)
        {
            clusters[init[i].cluster].push_back(init[i]);
        }

        ofile << "Itration: "<< itrations << endl;
        // Write points grouped by clusters
        for (unsigned int i = 0; i < k; i++)
        {
            ofile << "Cluster " << i + 1 << ": ";
            for (unsigned int j = 0; j < clusters[i].size(); j++)
            {
                ofile << "(" << clusters[i][j].x << "," << clusters[i][j].y << ") ";
            }
            ofile << "\n";
        }
        ofile << "\n";

        //Mean of clusters
        for (unsigned int i = 0; i < k; i++)
        {
            ofile << "Cluster " << i + 1 << " mean: ";
            ofile << "(" << centers[i].x << ", " << centers[i].y <<")";
            ofile << "\n";
        }
        ofile << "\n";

        // break if all centers converge within epsilon
        if (converged)
        {
            break;
        }

        // update previous centers
        prev_centers = centers;
    }

    ofile.close();
}
