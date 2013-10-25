/*

Simulation of flow inside a 2D square cavity
using the lattice Boltzmann method (LBM)

Written by:       Abhijit Joshi (abhijit@accelereyes.com)

Last modified on: Thursday, July 18 2013 @12:08 pm

Build instructions: make (uses Makefile present in this folder)

Run instructions: ./gpu_lbm

*/

#include<iostream>
#include<stdio.h>
#include<arrayfire.h>
using namespace af;

// problem parameters

const int     N = 128;                  // number of node points along X and Y (cavity length in lattice units)
const int     TIME_STEPS = 10000;       // number of time steps for which the simulation is run
const double  REYNOLDS_NUMBER = 1E6;    // REYNOLDS_NUMBER = LID_VELOCITY * N / kinematicViscosity

// don't change these unless you know what you are doing

const int     Q = 9;                    // number of discrete velocity aections used
const double  DENSITY = 2.7;            // fluid density in lattice units
const double  LID_VELOCITY = 0.05;      // lid velocity in lattice units

// allocate memory

// distribution functions
array     f(N, N, Q, f64);
array   feq(N, N, Q, f64);
array f_new(N, N, Q, f64);

// density and velocity
array   rho(N, N, f64);
array    ux(N, N, f64);
array    uy(N, N, f64);

// rate-of-strain
array sigma(N, N, f64);

// D3Q9 parameters
array    ex(Q, 1, f64);
array    ey(Q, 1, f64);
array oppos(Q, 1, u32);
array    wt(Q, 1, f64);

// D2Q9 parameters 

// populate D3Q19 parameters and copy them to __constant__ memory on the GPU

void D3Q9(array &ex, array &ey, array &oppos, array &wt)
{
    // D2Q9 model base velocities and weights

    ex(0,0) =  0.0;   ey(0,0) =  0.0;   wt(0,0) = 4.0 /  9.0;
    ex(1,0) =  1.0;   ey(1,0) =  0.0;   wt(1,0) = 1.0 /  9.0;
    ex(2,0) =  0.0;   ey(2,0) =  1.0;   wt(2,0) = 1.0 /  9.0;
    ex(3,0) = -1.0;   ey(3,0) =  0.0;   wt(3,0) = 1.0 /  9.0;
    ex(4,0) =  0.0;   ey(4,0) = -1.0;   wt(4,0) = 1.0 /  9.0;
    ex(5,0) =  1.0;   ey(5,0) =  1.0;   wt(5,0) = 1.0 / 36.0;
    ex(6,0) = -1.0;   ey(6,0) =  1.0;   wt(6,0) = 1.0 / 36.0;
    ex(7,0) = -1.0;   ey(7,0) = -1.0;   wt(7,0) = 1.0 / 36.0;
    ex(8,0) =  1.0;   ey(8,0) = -1.0;   wt(8,0) = 1.0 / 36.0;

    // define opposite (anti) aections (useful for implementing bounce back)

    oppos(0,0) = 0;      //      6        2        5
    oppos(1,0) = 3;      //               ^
    oppos(2,0) = 4;      //               |
    oppos(3,0) = 1;      //               |
    oppos(4,0) = 2;      //      3 <----- 0 -----> 1
    oppos(5,0) = 7;      //               |
    oppos(6,0) = 8;      //               |
    oppos(7,0) = 5;      //               v
    oppos(8,0) = 6;      //      7        4        8
}

// initialize values for aection vectors, density, velocity and distribution functions on the GPU

void initialize(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, 
                array &ex, array &ey, array &oppos, array &wt,
                array &rho, array &ux, array &uy, array &sigma, 
                array &f, array &feq, array &f_new)
{
    // loop over all voxels
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {

            // initialize density and velocity fields inside the cavity

              rho(i,j) = DENSITY;   // density
               ux(i,j) = 0.0;       // x-component of velocity
               uy(i,j) = 0.0;       // x-component of velocity
            sigma(i,j) = 0.0;       // rate-of-strain field

            // specify boundary condition for the moving lid

            if(j==0) ux(i,0) = LID_VELOCITY;

            // assign initial values for distribution functions
            // along various aections using equilibriu, functions

            for(int a=0;a<Q;a++) {
        
                array edotu = ex(a,1)*ux(i,j) + ey(a,1)*uy(i,j);
                array udotu = ux(i,j)*ux(i,j) + uy(i,j)*uy(i,j);

                feq(i,j,a)   = rho(i,j) * wt(a,0) * (1.0 + 3.0*edotu + 4.5*edotu*edotu - 1.5*udotu);
                f(i,j,a)     = feq(i,j,a);
                f_new(i,j,a) = feq(i,j,a);

            }

        }
    }
}


// this function updates the values of the distribution functions at all points along all directions
// carries out one lattice time-step (streaming + collision) in the algorithm
/*
void collideAndStream(// READ-ONLY parameters (used by this function but not changed)
                                 const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,
                                 const array &ex, const array &ey, const array &oppos, const array &wt,
                                 // READ + WRITE parameters (get updated in this function)
                                 array &rho,        // density
                                 array &ux,         // X-velocity
                                 array &uy,         // Y-velocity
                                 array &sigma,      // rate-of-strain
                                 array &f,          // distribution function
                                 array &feq,        // equilibrium distribution function
                                 array &f_new)      // new distribution function
{
    // loop over all interior voxels
    for(int i = 1; i < N-1; i++) {
        for(int j = 1; j < N-1; j++) {

    // natural index
    int index = i*N + j;  // column-major ordering

    // calculate fluid viscosity based on the Reynolds number
    double kinematicViscosity = LID_VELOCITY * (double) N / REYNOLDS_NUMBER;

    // calculate relaxation time tau
    double tau =  0.5 + 3.0 * kinematicViscosity;

    // collision
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        double edotu = ex[a]*ux[index] + ey[a]*uy[index];
        double udotu = ux[index]*ux[index] + uy[index]*uy[index];
        feq[index_f] = rho[index] * wt[a] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
    }

    // streaming from interior node points

    for(int a=0;a<Q;a++) {

        int index_f = a + index*Q;
        int index_nbr = (i+ex[a])*N + (j+ey[a]);
        int index_nbr_f = a + index_nbr * Q;
        int indexoppos = oppos[a] + index*Q;

        double tau_eff, tau_t, C_Smagorinsky;  // turbulence model parameters

        C_Smagorinsky = 0.16;

        // tau_t = additional contribution to the relaxation time 
        //         because of the "eddy viscosity" model
        // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        // REFERENCE: Krafczyk M., Tolke J. and Luo L.-S. (2003)
        //            Large-Eddy Simulations with a Multiple-Relaxation-Time LBE Model
        //            International Journal of Modern Physics B, Vol.17, 33-39
        // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        tau_t = 0.5*(pow(pow(tau,2) + 18.0*pow(C_Smagorinsky,2)*sigma[index],0.5) - tau);

        // the effective relaxation time accounts for the additional "eddy viscosity"
        // effects. Note that tau_eff now varies from point to point in the domain, and is
        // larger for large strain rates. If the strain rate is zero, tau_eff = 0 and we
        // revert back to the original (laminar) LBM scheme where tau_eff = tau.

        tau_eff = tau + tau_t;

        // post-collision distribution at (i,j) along "a"
        double f_plus = f[index_f] - (f[index_f] - feq[index_f])/tau_eff;

        int iS = i + ex[a]; int jS = j + ey[a];

        if((iS==0) || (iS==N-1) || (jS==0) || (jS==N-1)) {
            // bounce back
            double ubdote = ux[index_nbr]*ex[a] + uy[index_nbr]*ey[a];
            f_new[indexoppos] = f_plus - 6.0 * DENSITY * wt[a] * ubdote;
        }
        else {
            // stream to neighbor
            f_new[index_nbr_f] = f_plus;
        }
    }

        } // j
    }//i
}

void everythingElse( // READ-ONLY parameters (used by this function but not changed)
                                const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,
                                 const double *ex, const double *ey, const int *oppos, const double *wt,
                                // READ + WRITE parameters (get updated in this function)
                                double *rho,         // density
                                double *ux,         // X-velocity
                                double *uy,         // Y-velocity
                                double *sigma,      // rate-of-strain
                                double *f,          // distribution function
                                double *feq,        // equilibrium distribution function
                                double *f_new)      // new distribution function
{
    // loop over all interior voxels
    for(int i = 1; i < N-1; i++) {
        for(int j = 1; j < N-1; j++) {

    // natural index
    int index = i*N + j;  // column-major ordering

    // push f_new into f
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        f[index_f] = f_new[index_f];
    }

    // update density at interior nodes
    rho[index]=0.0;
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        rho[index] += f_new[index_f];
    }

    // update velocity at interior nodes
    double velx=0.0;
    double vely=0.0;
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        velx += f_new[index_f]*ex[a];
        vely += f_new[index_f]*ey[a];
    }
    ux[index] = velx/rho[index];
    uy[index] = vely/rho[index];

    // update the rate-of-strain field
    double sum_xx = 0.0, sum_xy = 0.0, sum_xz = 0.0;
    double sum_yx = 0.0, sum_yy = 0.0, sum_yz = 0.0;
    double sum_zx = 0.0, sum_zy = 0.0, sum_zz = 0.0;
    for(int a=1; a<Q; a++)
    {
        int index_f = a + index*Q;

        sum_xx = sum_xx + (f_new[index_f] - feq[index_f])*ex[a]*ex[a];
        sum_xy = sum_xy + (f_new[index_f] - feq[index_f])*ex[a]*ey[a];
        sum_xz = 0.0;
        sum_yx = sum_xy;
        sum_yy = sum_yy + (f_new[index_f] - feq[index_f])*ey[a]*ey[a];
        sum_yz = 0.0;
        sum_zx = 0.0;
        sum_zy = 0.0;
        sum_zz = 0.0;
    }

    // evaluate |S| (magnitude of the strain-rate)
    sigma[index] = pow(sum_xx,2) + pow(sum_xy,2) + pow(sum_xz,2)
                 + pow(sum_yx,2) + pow(sum_yy,2) + pow(sum_yz,2)
                 + pow(sum_zx,2) + pow(sum_zy,2) + pow(sum_zz,2);

    sigma[index] = pow(sigma[index],0.5);

        }//j
    }//i
}
*/
int main(int argc, char* argv[])
{
    try {

        // check whether to do graphics stuff or not
        bool isconsole = (argc == 2 && argv[1][0] == '-');

        // fill D3Q9 parameters in constant memory on the GPU
        D3Q9(ex, ey, oppos, wt);
/*
        // launch GPU kernel to initialize all fields
        initialize(N, Q, DENSITY, LID_VELOCITY, ex, ey, oppos, wt, rho, ux, uy, sigma, f, feq, f_new);
  
        // time integration
        int time=0;
        while(time<TIME_STEPS) {

            time++;

            std::cout << "Time = " << time << std::endl;

            collideAndStream(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER, ex, ey, oppos, wt, rho, ux, uy, sigma, f, feq, f_new);

            // collideAndStream and everythingElse were originally one kernel
            // they were separated out to make all threads synchronize globally
            // before moving on to the next set of calculations

            everythingElse(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER, ex, ey, oppos, wt, rho, ux, uy, sigma, f, feq, f_new);

            // this is where ArrayFire is currently used
            // the cool thing is you don't need to move the GPU arrays back to the
            // CPU for visualizing them. And of course, we have in-situ graphics

    //      double curl_min = 0, curl_max = 0;

            if (time % 10 == 0) {
                if(!isconsole) {
                    array umag = pow(ux*ux + uy*uy, 0.5);

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
*/
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
