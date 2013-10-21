/*

Simulation of flow inside a 2D square cavity
using the lattice Boltzmann method (LBM)

Written by:       Abhijit Joshi (abhijit@accelereyes.com)

Last modified on: Thursday, July 18 2013 @12:08 pm

Build instructions: make (uses Makefile present in this folder)

Run instructions: optirun ./gpu_lbm

*/

#include<iostream>
#include<stdio.h>
#include<arrayfire.h>
using namespace af;

// problem parameters

const int     N = 128;                  // number of node points along X and Y (cavity length in lattice units)
const int     TIME_STEPS = 1000000;     // number of time steps for which the simulation is run
const double  REYNOLDS_NUMBER = 1E4;    // REYNOLDS_NUMBER = LID_VELOCITY * N / kinematicViscosity

// don't change these unless you know what you are doing

const int     Q = 9;                    // number of discrete velocity directions used
const double  DENSITY = 2.7;            // fluid density in lattice units
const double  LID_VELOCITY = 0.05;      // lid velocity in lattice units

// initialize values for direction vectors, density, velocity and distribution functions on the GPU

__global__ void initialize(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, 
                           double *ex, double *ey, double *wt, int *oppos, 
                           double *rho, double *ux, double *uy, double* sigma, 
                           double *f, double *feq, double *f_new)
{
    // D2Q9 model base velocities and weights

    ex[0] =  0.0;   ey[0] =  0.0;   wt[0] = 4.0 /  9.0;
    ex[1] =  1.0;   ey[1] =  0.0;   wt[1] = 1.0 /  9.0;
    ex[2] =  0.0;   ey[2] =  1.0;   wt[2] = 1.0 /  9.0;
    ex[3] = -1.0;   ey[3] =  0.0;   wt[3] = 1.0 /  9.0;
    ex[4] =  0.0;   ey[4] = -1.0;   wt[4] = 1.0 /  9.0;
    ex[5] =  1.0;   ey[5] =  1.0;   wt[5] = 1.0 / 36.0;
    ex[6] = -1.0;   ey[6] =  1.0;   wt[6] = 1.0 / 36.0;
    ex[7] = -1.0;   ey[7] = -1.0;   wt[7] = 1.0 / 36.0;
    ex[8] =  1.0;   ey[8] = -1.0;   wt[8] = 1.0 / 36.0;

    // define opposite (anti) directions (useful for implementing bounce back)

    oppos[0] = 0;      //      6        2        5
    oppos[1] = 3;      //               ^
    oppos[2] = 4;      //               |
    oppos[3] = 1;      //               |
    oppos[4] = 2;      //      3 <----- 0 -----> 1
    oppos[5] = 7;      //               |
    oppos[6] = 8;      //               |
    oppos[7] = 5;      //               v
    oppos[8] = 6;      //      7        4        8

    // compute the global "i" and "j" location handled by this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    // natural index for location (i,j)

    int ixy = i*N+j;

    // initialize density and velocity fields inside the cavity

    rho[ixy] = DENSITY;
    ux[ixy] = 0.0;
    uy[ixy] = 0.0;
    sigma[ixy] = 0.0;

    // specify boundary condition for the moving lid

    if((i>0) && (i<N-1) &&(j==0)) ux[ixy] = LID_VELOCITY;

    // assign initial values for distribution functions
    // along various directions using equilibriu, functions

    for(int dir=0;dir<Q;dir++) {

        int index = i*N*Q + j*Q + dir;

        double edotu = ex[dir]*ux[ixy] + ey[dir]*uy[ixy];
        double udotu = ux[ixy]*ux[ixy] + uy[ixy]*uy[ixy];

        feq[index]   = rho[ixy] * wt[dir] * (1.0 + 3.0*edotu + 4.5*edotu*edotu - 1.5*udotu);
        f[index]     = feq[index];
        f_new[index] = feq[index];

    }
}

// this function updates the values of the distribution functions at all points along all directions
// carries out one lattice time-step (streaming + collision) in the algorithm

__global__ void collideAndStream( // READ-ONLY parameters (used by this function but not changed)
                                 const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,
                                 const double *ex,      // x-component of direction vector
                                 const double *ey,      // x-component of direction vector
                                 const double *wt,   // weight factor for each direction
                                 const int *oppos,        // anti (opposite) vector for each direction

                                 // READ + WRITE parameters (get updated in this function)

                                 double *rho,         // density
                                 double *ux,         // X-velocity
                                 double *uy,         // Y-velocity
                                 double *sigma,      // rate-of-strain
                                 double *f,          // distribution function
                                 double *feq,        // equilibrium distribution function
                                 double *f_new)      // new distribution function
{
    // calculate fluid viscosity based on the Reynolds number
    double kinematicViscosity = LID_VELOCITY * (double) N / REYNOLDS_NUMBER;


    // calculate relaxation time tau
    double tau =  0.5 + 3.0 * kinematicViscosity;

    // compute the global "i" and "j" location of this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    {
        // collision
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                int ixy = i*N + j;
                for(int dir=0;dir<Q;dir++) {
                    int index = i*N*Q + j*Q + dir;
                    double edotu = ex[dir]*ux[ixy] + ey[dir]*uy[ixy];
                    double udotu = ux[ixy]*ux[ixy] + uy[ixy]*uy[ixy];
                    feq[index] = rho[ixy] * wt[dir] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
                }
        }

        // streaming from interior node points

        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                for(int dir=0;dir<Q;dir++) {

                    int index = i*N*Q + j*Q + dir;
                    int index_new = (i+ex[dir])*N*Q + (j+ey[dir])*Q + dir;
                    int indexoppos = i*N*Q + j*Q + oppos[dir];

                    double tau_eff, tau_t, C_Smagorinsky;  // turbulence model parameters

                    C_Smagorinsky = 0.16;

                    // tau_t = additional contribution to the relaxation time 
                    //         because of the "eddy viscosity" model
                    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                    // REFERENCE: Krafczyk M., Tolke J. and Luo L.-S. (2003)
                    //            Large-Eddy Simulations with a Multiple-Relaxation-Time LBE Model
                    //            International Journal of Modern Physics B, Vol.17, 33-39
                    // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

                    tau_t = 0.5*(pow(pow(tau,2) + 18.0*pow(C_Smagorinsky,2)*sigma[i*N+j],0.5) - tau);

//                  if((i==250) && (j==1)) printf("tau_t = %f\n",tau_t);

                    // the effective relaxation time accounts for the additional "eddy viscosity"
                    // effects. Note that tau_eff now varies from point to point in the domain, and is
                    // larger for large strain rates. If the strain rate is zero, tau_eff = 0 and we
                    // revert back to the original (laminar) LBM scheme where tau_eff = tau.

                    tau_eff = tau + tau_t;

                    // post-collision distribution at (i,j) along "dir"
                    double f_plus = f[index] - (f[index] - feq[index])/tau_eff;

//                  if ((i==250) && (j==1)) printf("j = %d dir = %d   f_plus = %f\n",j,dir,f_plus);

                    int iS = i + ex[dir]; int jS = j + ey[dir];

                    if((iS==0) || (iS==N-1) || (jS==0) || (jS==N-1)) {
                        // bounce back
                        int ixy_nbr = iS*N + jS;
                        double ubdote = ux[ixy_nbr]*ex[dir] + uy[ixy_nbr]*ey[dir];
                        f_new[indexoppos] = f_plus - 6.0 * DENSITY * wt[dir] * ubdote;

//                      if((i==250) && (j+ey[dir]==0)) {
//                        printf("ubdote = %f  Bounce backed f(%d) = %f\n",ubdote, oppos[dir],f_new[indexoppos]);
//                      }


                    }
                    else {
                        // stream to neighbor
                        f_new[index_new] = f_plus;
                    }
                }
        }
    }
}

__global__ void everythingElse( // READ-ONLY parameters (used by this function but not changed)
                                 const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,
                                 const double *ex,      // x-component of direction vector
                                 const double *ey,      // x-component of direction vector
                                 const double *wt,   // weight factor for each direction
                                 const int *oppos,        // anti (opposite) vector for each direction

                                 // READ + WRITE parameters (get updated in this function)

                                 double *rho,         // density
                                 double *ux,         // X-velocity
                                 double *uy,         // Y-velocity
                                 double *sigma,      // rate-of-strain
                                 double *f,          // distribution function
                                 double *feq,        // equilibrium distribution function
                                 double *f_new)      // new distribution function
{
    // compute the global "i" and "j" location of this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    {
        // push f_new into f
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                for(int dir=0;dir<Q;dir++) {
                    int index = i*N*Q + j*Q + dir;
                    f[index] = f_new[index];
//                  if ((i==250) && (j==1)) printf("j = %d dir = %d   f_new = %f\n",j,dir,f_new[index]);
                }
        }

        // update density at interior nodes
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                rho[i*N+j]=0.0;
                for(int dir=0;dir<Q;dir++) {
                    int index = i*N*Q + j*Q + dir;
                    rho[i*N+j] += f_new[index];
//                  if ((i==250) && (j==1)) printf("f(%d) = %f\n",dir,f_new[index]);
                }
        }

        // update velocity at interior nodes
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                double velx=0.0;
                double vely=0.0;
                for(int dir=0;dir<Q;dir++) {
                    int index = i*N*Q + j*Q + dir;
                    velx+=f_new[index]*ex[dir];
                    vely+=f_new[index]*ey[dir];
                }
                int ixy = i*N+j;
                ux[ixy] = velx/rho[ixy] ;
                uy[ixy] = vely/rho[ixy] ;
        }

        // update the rate-of-strain field
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
            double sum_xx = 0.0, sum_xy = 0.0, sum_xz = 0.0;
            double sum_yx = 0.0, sum_yy = 0.0, sum_yz = 0.0;
            double sum_zx = 0.0, sum_zy = 0.0, sum_zz = 0.0;
            for(int dir=1; dir<Q; dir++)
            {
              int index = i*N*Q + j*Q + dir;

              sum_xx = sum_xx + (f_new[index] - feq[index])*ex[dir]*ex[dir];
              sum_xy = sum_xy + (f_new[index] - feq[index])*ex[dir]*ey[dir];
              sum_xz = 0.0;
              sum_yx = sum_xy;
              sum_yy = sum_yy + (f_new[index] - feq[index])*ey[dir]*ey[dir];
              sum_yz = 0.0;
              sum_zx = 0.0;
              sum_zy = 0.0;
              sum_zz = 0.0;
            }

            // evaluate |S| (magnitude of the strain-rate)
            int i_j    = i*N+j;

            sigma[i_j] = pow(sum_xx,2) + pow(sum_xy,2) + pow(sum_xz,2)
                        + pow(sum_yx,2) + pow(sum_yy,2) + pow(sum_yz,2)
                        + pow(sum_zx,2) + pow(sum_zy,2) + pow(sum_zz,2);

            sigma[i_j] = pow(sigma[i_j],0.5);

        }

    }
}

int main(int argc, char* argv[])
{
    try {

        // check whether to do graphics stuff or not
        bool isconsole = (argc == 2 && argv[1][0] == '-');

        // allocate memory on the GPU

        // the base vectors and associated weight coefficients (GPU)
        double *ex, *ey, *wt;  // pointers to device (GPU) memory
        cudaMalloc((void **)&ex,Q*sizeof(double));
        cudaMalloc((void **)&ey,Q*sizeof(double));
        cudaMalloc((void **)&wt,Q*sizeof(double));

        // ant vector (GPU)
        int *oppos;  // gpu memory
        cudaMalloc((void **)&oppos,Q*sizeof(int));

        // distribution functions
        double *f, *feq, *f_new;
        cudaMalloc((void **)&f,N*N*Q*sizeof(double));
        cudaMalloc((void **)&feq,N*N*Q*sizeof(double));
        cudaMalloc((void **)&f_new,N*N*Q*sizeof(double));

        // density and velocity
        double *rho, *ux, *uy;
        cudaMalloc((void **)&rho,N*N*sizeof(double));
        cudaMalloc((void **)&ux,N*N*sizeof(double));
        cudaMalloc((void **)&uy,N*N*sizeof(double));

        // rate-of-strain
        double *sigma;
        cudaMalloc((void **)&sigma,N*N*sizeof(double));

        // assign a 2D distribution of CUDA "threads" within each CUDA "block"    
        int threadsAlongX=16, threadsAlongY=16;
        dim3 dimBlock(threadsAlongX, threadsAlongY, 1);

        // calculate number of blocks along X and Y in a 2D CUDA "grid"
        dim3 dimGrid( ceil(float(N)/float(dimBlock.x)), ceil(float(N)/float(dimBlock.y)), 1 );

        // launch GPU kernel to initialize all fields
        initialize<<<dimGrid,dimBlock>>>(N, Q, DENSITY, LID_VELOCITY,
                                         ex, ey, wt, oppos,
                                         rho, ux, uy, sigma,
                                         f, feq, f_new);

        // time integration
        int time=0;
        while(time<TIME_STEPS) {

            time++;

            std::cout << "Time = " << time << std::endl;

            collideAndStream<<<dimGrid,dimBlock >>>(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER,
                                                    ex, ey, wt, oppos,
                                                    rho, ux, uy, sigma,
                                                    f, feq, f_new);

            // collideAndStream and everythingElse were originally one kernel
            // they were separated out to make all threads synchronize globally
            // before moving on to the next set of calculations

            everythingElse<<<dimGrid,dimBlock >>>(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER,
                                                  ex, ey, wt, oppos,
                                                  rho, ux, uy, sigma,
                                                  f, feq, f_new);

            // this is where ArrayFire is currently used
            // the cool thing is you don't need to move the GPU arrays back to the
            // CPU for visualizing them. And of course, we have in-situ graphics

    //      double curl_min = 0, curl_max = 0;

            if (time % 10 == 0) {
                if(!isconsole) {
                    array U(N,N,ux,afDevice);
                    array V(N,N,uy,afDevice);
                    array umag = pow(U*U + V*V, 0.5);

//                  array dUdx,dUdy,dVdx,dVdy;
//                  grad(dUdx,dUdy,U);
//                  grad(dVdx,dVdy,V);
//                  array curl = dVdx - dUdy;

//                  double2 extrema = minmax<double2>(curl);
//                  std::cout << "Curl --- min " << extrema.x << "  max " << extrema.y << std::endl;

//                  if (extrema.x < curl_min) curl_min = extrema.x;
//                  if (extrema.y > curl_max) curl_max = extrema.y;

//                  curl(0) = -0.1;
//                  curl(N) = +0.1;
                    fig("color","heat");
                    image(umag);
                }
            }
    
        }

    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
