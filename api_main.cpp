#include <iostream>
#include <sstream>
#include <cassert>

#include <algorithm>


#include "Vertex.h"
#include "Distance_ANN.h"
#include "Core.h"
#include "Cluster_Basic.h"
#include "Point.h"
#include "Density.h"


using namespace std;


//rename for brevity
typedef Vertex<ANNPoint,Cluster_Info > Point;

// comparison function object for vector indices
template<class V> class Less_Than {
  protected:
  V& v;
  public:
  Less_Than (V& v_): v(v_){}
  bool operator()(const int a, const int  b) const 
  {return Point::Less_Than()(v[a], v[b]);}
};


extern "C" void CCLST(double* matrix, int rows, int cols, double rips, double ttau) {
  vector<double> dataVec;
  // or the std::vector(row,row+n)
  dataVec.insert(dataVec.end(), &matrix[0], &matrix[rows*cols]);
  CreateClust(dataVec,cols,rips,ttau);
  // now offload payload
}

extern "C" void CDCLST(double* matrix, int rows, int cols, double rips, double ttau, double nns) {
  vector<double> dataVec;
  // or the std::vector(row,row+n)
  dataVec.insert(dataVec.end(), &matrix[0], &matrix[rows*cols]);
  CreateDensityClust(dataVec,cols,rips,ttau,nns);
  // now offload payload
}

void CreateClust(vector<double> row, int cols, double r, double ttau) {
  int dim = -1;
  int nb_points = 0;
  vector< Point > point_cloud;
  
  for (int i=0; i<cols; i++) {
    double d;
    dim = row.size()-1;
    ANNPoint p(dim);
    p.coord = new double[dim];
    for (int j=0; j<dim; j++)
      p.coord[j] = row[j];
    Point v(p);
    v.set_func(row[dim]);
    v.data.boundary_flag=false;
    point_cloud.push_back(v);
    nb_points++;
  }

  vector<int> perm;
  perm.reserve(nb_points);
  for(int i=0; i < nb_points; i++)
    perm.push_back(i);
  std::sort(perm.begin(), perm.end(), Less_Than<vector<Point> >(point_cloud));
  // store inverse permutation as array of iterators on initial point cloud
  vector< vector<Point>::iterator> pperm;
  pperm.reserve(nb_points);
  for (int i=0; i<nb_points; i++)
    pperm.push_back(point_cloud.begin());
  for (int i=0; i<nb_points; i++)
    pperm[perm[i]] = (point_cloud.begin() + i);
  // operate permutation on initial point cloud 
  vector<Point> pc;
  pc.reserve(nb_points);
  for (int i=0; i<nb_points; i++)
    pc.push_back(point_cloud[i]);
  for (int i=0; i<nb_points; i++)
    point_cloud[i] = pc[perm[i]];

  // create distance structure
  Distance_ANN< vector< Point >::iterator > metric_information;

  metric_information.initialize(point_cloud.begin(),
				point_cloud.end(),
				dim);
  metric_information.mu = r*r;
  Cluster< vector< Point >::iterator > output_clusters;
  output_clusters.tau = ttau;
  // perform clustering
  compute_persistence(point_cloud.begin(),point_cloud.end(),
  	      metric_information,output_clusters);


  // compress data structure:
  // attach each data point to its cluster's root directly
  // to speed up output processing
  attach_to_clusterheads(point_cloud.begin(),point_cloud.end());

  // this needs to be output as some kind of struct.
  // output clusters (use permutation to preserve original point order)
  ofstream out;
  out.open("clusters.txt");
  output_clusters.output_clusters(out, pperm.begin(), pperm.end());
  out.close();

  //output barcode
  out.open("diagram.txt");
  output_clusters.output_intervals(out);
  out.close();
  
  //output colored clusters to COFF file (first 3 dimensions are selected)
  out.open("clusters_3d.coff");
  output_clusters.output_clusters_coff(out,point_cloud.begin(),point_cloud.end());
  out.close();

}

void CreateDensityCluster(vector<double> row, int cols, double r, double ttau, int nns) {
  int com=1;
  vector< Point > point_cloud;
  int dim = -1;
  int nb_points = 0;
  for (int i=0; i<cols; i++) {
    double d;


  }  
}
