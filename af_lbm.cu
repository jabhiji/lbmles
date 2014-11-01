/*

Simulation of flow inside a 2D square cavity
using the lattice Boltzmann method (LBM)

Written by:       Abhijit Joshi (abhijit@accelereyes.com)

Last modified on: Thursday, July 18 2013 @12:08 pm

Build instructions: make (uses Makefile present in this folder)

Run instructions: ./gpu_lbm //slower than cpu_version

*/

#include<iostream>
#include<stdio.h>
#include<arrayfire.h>
using namespace af;

// problem parameters

const int     N = 8;                  // number of node points along X and Y (cavity length in lattice units)
const int     TIME_STEPS = 1;         // number of time steps for which the simulation is run
const double  REYNOLDS_NUMBER = 1E2;  // REYNOLDS_NUMBER = LID_VELOCITY * N / kinematicViscosity

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

// populate D3Q19 parameters and copy them to __constant__ memory on the GPU

void D3Q9(double *ex, double *ey, int *oppos, double *wt)
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

    // define opposite [anti] aections [useful for implementing bounce back]

    oppos[0] = 0;      //      6        2        5
    oppos[1] = 3;      //               ^
    oppos[2] = 4;      //               |
    oppos[3] = 1;      //               |
    oppos[4] = 2;      //      3 <----- 0 -----> 1
    oppos[5] = 7;      //               |
    oppos[6] = 8;      //               |
    oppos[7] = 5;      //               v
    oppos[8] = 6;      //      7        4        8
}

// initialize values for direction vectors, density, velocity and distribution functions on the GPU

void initialize(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, 
                double *ex_h, double *ey_h, int *op_h, double *wt_h,
                array &rho, array &ux, array &uy, array &sigma, 
                array &f, array &feq, array &f_new)
{
    // populate D2Q9 buffers
    array ex(Q,ex_h,afHost,f64);
    array ey(Q,ey_h,afHost,f64);
    array op(Q,op_h,afHost,u32);
    array wt(Q,wt_h,afHost,f64);

    print(wt);

    // initialize 2D fields
    rho = constant(DENSITY,N,N,f64);   // density
    ux = constant(0.0,N,N,f64);        // x-component of velocity
    uy = constant(0.0,N,N,f64);        // x-component of velocity
    sigma = constant(0.0,N,N,f64);     // rate-of-strain

    ux(0,span) = LID_VELOCITY;  // moving lid on top

    print(rho);
    print(ux);
    print(uy);

    for(int a=0;a<Q;a++) {
    for(int j=0;j<N;j++) {
    for(int i=0;i<N;i++) {
        array edotu = ux(i,j)*ex(a) + uy(i,j)*ey(a);
        array udotu = ux(i,j)*ux(i,j) + uy(i,j)*uy(i,j);
        feq(i,j,a) = rho(i,j) * wt_h[a]; // * (1.0 + 3.0*edotu + 4.5*edotu*edotu - 1.5*udotu); 
    }
    }
    }
  
    f = feq;
    f_new = feq;

    print(feq);
}


// this function updates the values of the distribution functions at all points along all directions
// carries out one lattice time-step (streaming + collision) in the algorithm

void collideAndStream(// READ-ONLY parameters (used by this function but not changed)
                                 const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,
                                 const double *ex_h, const double *ey_h, const int *op_h, const double *wt_h,
                                 // READ + WRITE parameters (get updated in this function)
                                 array &rho,        // density
                                 array &ux,         // X-velocity
                                 array &uy,         // Y-velocity
                                 array &sigma,      // rate-of-strain
                                 array &f,          // distribution function
                                 array &feq,        // equilibrium distribution function
                                 array &f_new)      // new distribution function
{
    // populate D2Q9 buffers
    array ex(Q,ex_h,afHost,f64);
    array ey(Q,ey_h,afHost,f64);
    array op(Q,op_h,afHost,u32);
    array wt(Q,wt_h,afHost,f64);

    // calculate fluid viscosity based on the Reynolds number
    double kinematicViscosity = LID_VELOCITY * (double) N / REYNOLDS_NUMBER;

    // calculate relaxation time tau
    array tau =  constant(0.5 + 3.0 * kinematicViscosity, N, N, f64);

    array edotu(N,N,Q,f64);
    array udotu(N,N,f64);

    // collision
    for(int a=0;a<Q;a++) {
        edotu(seq(1,end-1),seq(1,end-1),a) = ex(a)*ux(seq(1,end-1), seq(1,end-1))   + ey(a)*uy(seq(1,end-1), seq(1,end-1));
        udotu(seq(1,end-1),seq(1,end-1)) = ux(seq(1,end-1), seq(1,end-1))*ux(seq(1,end-1), seq(1,end-1)) + uy(seq(1,end-1), seq(1,end-1))*uy(seq(1,end-1), seq(1,end-1));
        feq(seq(1,end-1), seq(1,end-1), a) = rho(seq(1,end-1), seq(1,end-1)) * wt(a) * (1 + 3*edotu(seq(1,end-1),seq(1,end-1),a) + 4.5*edotu(seq(1,end-1),seq(1,end-1),a)*edotu(seq(1,end-1),seq(1,end-1),a) - 1.5*udotu(seq(1,end-1),seq(1,end-1)));
    }

    // streaming from interior node points

    array tau_t(N,N,f64);
    array tau_eff(N,N,f64);
    array f_plus(N,N,Q,f64);

    const double C_Smagorinsky = 0.16;  // turbulence model parameters

    // local relaxation time depends on the rate-of-strain
    tau_t(seq(1,end-1), seq(1,end-1)) = 0.5*(  pow( pow(tau(seq(1,end-1), seq(1,end-1)),2) 
                                                 + 18.0*pow(C_Smagorinsky,2)*sigma(seq(1,end-1), seq(1,end-1)),0.5) 
                                                 - tau(seq(1,end-1), seq(1,end-1)));

    tau_eff = tau + tau_t;

    for(int a=0;a<Q;a++) {

        // post-collision distribution at (i,j) along all directions
        f_plus(seq(1,end-1), seq(1,end-1), a) =    f(seq(1,end-1), seq(1,end-1), a)
                                              - (  f(seq(1,end-1), seq(1,end-1), a) - 
                                                 feq(seq(1,end-1), seq(1,end-1), a))
                                              / tau_eff(seq(1,end-1), seq(1,end-1));
    }

    for(int i = 1; i < N-1; i++) {
        for(int j = 1; j < N-1; j++) {
            for(int a = 0; a < Q; a++) {

                int iS = i + ex_h[a]; int jS = j + ey_h[a];
  
                if((iS==0) || (iS==N-1) || (jS==0) || (jS==N-1)) {
                    // bounce back
                    array ubdote = ux(iS,jS)*ex(a) + uy(iS,jS)*ey(a);
                    f_new(i,j,op_h[a]) = f_plus(i,j,a) - 6.0 * DENSITY * wt(a) * ubdote;
                }
                else {
                    // stream to neighbor
                    f_new(iS,jS,a) = f_plus(i,j,a);
                }

           }
        }
    }

}


void everythingElse( // READ-ONLY parameters (used by this function but not changed)
                                const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,
                                const double *ex_h, const double *ey_h, const int *op_h, const double *wt_h,
                                // READ + WRITE parameters (get updated in this function)
                                array &rho,         // density
                                array &ux,         // X-velocity
                                array &uy,         // Y-velocity
                                array &sigma,      // rate-of-strain
                                array &f,          // distribution function
                                array &feq,        // equilibrium distribution function
                                array &f_new)      // new distribution function
{
    // populate D2Q9 buffers
    array ex(Q,ex_h,afHost,f64);
    array ey(Q,ey_h,afHost,f64);
    array op(Q,op_h,afHost,u32);
    array wt(Q,wt_h,afHost,f64);

    // push f_new into f
    f = f_new;

    // update density at interior nodes
    rho = sum(f,2);

    // update velocity at interior nodes
    array velx=constant(0.0,N,N,f64);
    array vely=constant(0.0,N,N,f64);
    for(int a=0;a<Q;a++) {
        velx(seq(1,end-1), seq(1,end-1)) = velx(seq(1,end-1), seq(1,end-1)) + f_new(seq(1,end-1), seq(1,end-1), a) * ex(a);
        vely(seq(1,end-1), seq(1,end-1)) = vely(seq(1,end-1), seq(1,end-1)) + f_new(seq(1,end-1), seq(1,end-1), a) * ey(a);
    }
    ux(seq(1,end-1), seq(1,end-1)) = velx(seq(1,end-1), seq(1,end-1))/rho(seq(1,end-1), seq(1,end-1));
    uy(seq(1,end-1), seq(1,end-1)) = vely(seq(1,end-1), seq(1,end-1))/rho(seq(1,end-1), seq(1,end-1));
  
    // update the rate-of-strain field
    array sum_xx = constant(0.0,N,N,f64); array sum_xy = constant(0.0,N,N,f64); array sum_xz = constant(0.0,N,N,f64);
    array sum_yx = constant(0.0,N,N,f64); array sum_yy = constant(0.0,N,N,f64); array sum_yz = constant(0.0,N,N,f64);
    array sum_zx = constant(0.0,N,N,f64); array sum_zy = constant(0.0,N,N,f64); array sum_zz = constant(0.0,N,N,f64);
  
    for(int a=1; a<Q; a++)
    {
        sum_xx(seq(1,end-1),seq(1,end-1)) = sum_xx(seq(1,end-1),seq(1,end-1)) + (f_new(seq(1,end-1),seq(1,end-1),a) - feq(seq(1,end-1),seq(1,end-1),a))*ex(a)*ex(a);
        sum_xy(seq(1,end-1),seq(1,end-1)) = sum_xy(seq(1,end-1),seq(1,end-1)) + (f_new(seq(1,end-1),seq(1,end-1),a) - feq(seq(1,end-1),seq(1,end-1),a))*ex(a)*ey(a);
        sum_xz(seq(1,end-1),seq(1,end-1)) = 0.0;
        sum_yx(seq(1,end-1),seq(1,end-1)) = sum_xy(seq(1,end-1),seq(1,end-1));
        sum_yy(seq(1,end-1),seq(1,end-1)) = sum_yy(seq(1,end-1),seq(1,end-1)) + (f_new(seq(1,end-1),seq(1,end-1),a) - feq(seq(1,end-1),seq(1,end-1),a))*ey(a)*ey(a);
        sum_yz(seq(1,end-1),seq(1,end-1)) = 0.0;
        sum_zx(seq(1,end-1),seq(1,end-1)) = 0.0;
        sum_zy(seq(1,end-1),seq(1,end-1)) = 0.0;
        sum_zz(seq(1,end-1),seq(1,end-1)) = 0.0;
    }
  
    // evaluate |S| (magnitude of the strain-rate)
    sigma(seq(1,end-1),seq(1,end-1)) = sum_xx(seq(1,end-1),seq(1,end-1)) * sum_xx(seq(1,end-1),seq(1,end-1))
                                     + sum_xy(seq(1,end-1),seq(1,end-1)) * sum_xy(seq(1,end-1),seq(1,end-1))
                                     + sum_xz(seq(1,end-1),seq(1,end-1)) * sum_xz(seq(1,end-1),seq(1,end-1))
                                     + sum_yx(seq(1,end-1),seq(1,end-1)) * sum_yx(seq(1,end-1),seq(1,end-1))
                                     + sum_yy(seq(1,end-1),seq(1,end-1)) * sum_yy(seq(1,end-1),seq(1,end-1))
                                     + sum_yz(seq(1,end-1),seq(1,end-1)) * sum_yz(seq(1,end-1),seq(1,end-1))
                                     + sum_zx(seq(1,end-1),seq(1,end-1)) * sum_zx(seq(1,end-1),seq(1,end-1))
                                     + sum_zy(seq(1,end-1),seq(1,end-1)) * sum_zy(seq(1,end-1),seq(1,end-1))
                                     + sum_zz(seq(1,end-1),seq(1,end-1)) * sum_zz(seq(1,end-1),seq(1,end-1));

    sigma(seq(1,end-1),seq(1,end-1)) = pow(sigma(seq(1,end-1),seq(1,end-1)),0.5);
}

int main(int argc, char* argv[])
{
    try {

        // check whether to do graphics stuff or not
        bool isconsole = (argc == 2 && argv[1][0] == '-');

        // D3Q9 parameters
        double *ex = new double[Q];
        double *ey = new double[Q];
        int    *op = new int[Q];
        double *wt = new double[Q];

        // fill D3Q9 parameters in constant memory on the GPU
        D3Q9(ex, ey, op, wt);

        // launch GPU kernel to initialize all fields
        initialize(N, Q, DENSITY, LID_VELOCITY, ex, ey, op, wt, rho, ux, uy, sigma, f, feq, f_new);
/*
        // time integration
        int time=0;
        while(time<TIME_STEPS) {

            time++;

            std::cout << "Time = " << time << std::endl;

            collideAndStream(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER, ex, ey, op, wt, rho, ux, uy, sigma, f, feq, f_new);

            // collideAndStream and everythingElse were originally one kernel
            // they were separated out to make all threads synchronize globally
            // before moving on to the next set of calculations

            everythingElse(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER, ex, ey, op, wt, rho, ux, uy, sigma, f, feq, f_new);


            if (time % 10 == 0) {
                if(!isconsole) {
                    array umag = pow(ux*ux + uy*uy, 0.5);
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
