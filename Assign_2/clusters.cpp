#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
using namespace std;

// representation of point;
// x,y - coordinates on XY
// cluster - assigned cluster, by default -1(not assigned)
struct point{
    double x, y;
    int cluster = -1;
};

int main(){
    int k;
    double epsilon;
    string filename;
    cin >> k >> epsilon >> filename;
    
    vector<point> points;
    ifstream file(filename);
    point tmp;
    while (file >> tmp.x >> tmp.y) points.push_back(tmp);
    file.close();

    vector<point> centers(k), prev_centers(k);
    vector<int> chosen_indices;
    while (chosen_indices.size() < k) {
        int idx = rand() % points.size();
        if (find(chosen_indices.begin(), chosen_indices.end(), idx) == chosen_indices.end()) {
            centers[chosen_indices.size()] = points[idx];
            chosen_indices.push_back(idx);
        }
    }

    ofstream ofile("out.txt");
    ofile << "Initial cluster centers:\n";
    for (int i = 0; i < k; i++)
        ofile << "Cluster " << i + 1 << " mean: (" << centers[i].x << ", " << centers[i].y << ")\n\n";

    int iterations = 0;
    while (true) {
        iterations++;
        prev_centers = centers;
        
        // Assign points to clusters
        for (auto& p : points) {
            double min_dist = INFINITY;
            for (int i = 0; i < k; i++) {
                double dist = hypot(p.x - centers[i].x, p.y - centers[i].y);
                if (dist < min_dist) {
                    min_dist = dist;
                    p.cluster = i;
                }
            }
        }

        // Calculate new centers
        vector<double> sum_x(k, 0), sum_y(k, 0);
        vector<int> count(k, 0);
        for (const auto& p : points) {
            sum_x[p.cluster] += p.x;
            sum_y[p.cluster] += p.y;
            count[p.cluster]++;
        }
        for (int i = 0; i < k; i++) {
            if (count[i] > 0) {
                centers[i].x = sum_x[i] / count[i];
                centers[i].y = sum_y[i] / count[i];
            }
        }

        // Output current state
        ofile << "Iteration: " << iterations << "\n";
        vector<vector<point>> clusters(k);
        for (const auto& p : points) clusters[p.cluster].push_back(p);
        
        for (int i = 0; i < k; i++) {
            ofile << "Cluster " << i + 1 << ": ";
            for (const auto& p : clusters[i])
                ofile << "(" << p.x << "," << p.y << ") ";
            ofile << "\n";
        }
        ofile << "\n";

        // Check convergence
        bool converged = true;
        for (int i = 0; i < k; i++) {
            if (hypot(centers[i].x - prev_centers[i].x, centers[i].y - prev_centers[i].y) >= epsilon) {
                converged = false;
                break;
            }
        }
        if (converged) break;
    }

    ofile.close();
    return 0;
}
