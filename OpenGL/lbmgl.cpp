// OpenGL specific headers

#include <GLFW/glfw3.h>

// the usual gang of C++ headers

#include <iostream>
#include <cmath>
#include <cstdlib>
#include<ctime>        // clock_t, clock(), CLOCKS_PER_SEC

// problem parameters

const int     N = 128;                  // number of node points along X and Y (cavity length in lattice units)
const int     TIME_STEPS = 10000000;    // number of time steps for which the simulation is run
const double  REYNOLDS_NUMBER = 1E6;    // REYNOLDS_NUMBER = LID_VELOCITY * N / kinematicViscosity

// don't change these unless you know what you are doing

const int     Q = 9;                    // number of discrete velocity aections used
const double  DENSITY = 2.7;            // fluid density in lattice units
const double  LID_VELOCITY = 0.05;      // lid velocity in lattice units

// calculate pixel colors for the current graphics window, defined by the
// minimum and maximum X and Y coordinates

void showVelocityField(GLFWwindow *window, int WIDTH, int HEIGHT, double xmin, double xmax, double ymin, double ymax, const double *ux, const double *uy)
{
    //--------------------------------
    //  OpenGL initialization stuff 
    //--------------------------------
    glfwGetFramebufferSize(window, &WIDTH, &HEIGHT);

    // select background color to be white
    // R = 1, G = 1, B = 1, alpha = 0
    glClearColor (1.0, 1.0, 1.0, 0.0);

    // initialize viewing values
    glMatrixMode(GL_PROJECTION);

    // replace current matrix with the identity matrix
    glLoadIdentity();

    // set clipping planes in the X-Y-Z coordinate system
    glOrtho(xmin,xmax,ymin,ymax, -1.0, 1.0);

    // clear all pixels
    glClear (GL_COLOR_BUFFER_BIT);

    // 2D array size is identical to the window size in pixels
    const int NX = WIDTH;
    const int NY = HEIGHT;

    // buffer actually used for plotting velocity magnitude
    float *umag = new float[WIDTH*HEIGHT];

    // fill the 2D array with the velocity magnitude
    for(int i = 0; i < NX; i++) {
      for(int j = 0; j < NY; j++) {

        // map pixel coordinate (i,j) to LBM lattice coordinates (x,y)
	int x = i*N/NX;
	int y = j*N/NY;
	int idx = x*N+y; 
        
        // calculate velocity magnitude buffer 
        umag[i*WIDTH + j] = pow((ux[idx]*ux[idx] + uy[idx]*uy[idx]), 0.5);
      }
    }
    
    // assign color based on the velocity magnitude - Red Green Blue (RGB) 

    float dx = (xmax - xmin)/NX;
    float dy = (ymax - ymin)/NY;

    for(int i = 0; i < NX; i++) {
        for(int j = 0; j < NY; j++) {

            float x = xmin + i*dx;   // actual x coordinate
            float y = ymin + j*dy;   // actual y coordinate
            float VAL = umag[i*WIDTH + j]/LID_VELOCITY;

            glColor3f(VAL,VAL,VAL);   // black(0,0,0) to white(1,1,1) transition
//          glColor3f(VAL,1-VAL,0);   // green(0,1,0) to red(1,0,0) transition
//          glColor3f(VAL,0,1-VAL);   // blue(0,0,1) to red(1,0,0) transition

glRectf (x,y,x+dx,y+dy);
      }
    }

    // swap front and back buffers
    glfwSwapBuffers(window);

    // poll for and processs events
    glfwPollEvents();
}

// Entry point for the display routine

void displaySolution(GLFWwindow *window, int WIDTH, int HEIGHT, const double *ux, const double *uy)
{
    // specify initial window size in the X-Y plane
    double xmin = 0, xmax = N, ymin = 0, ymax = N;

    // display the Mandelbrot set in (xmin,ymin)-(xmax,ymax)
    showVelocityField(window, WIDTH, HEIGHT, xmin, xmax, ymin, ymax, ux, uy);
}

/*
int displaySolution(const double *ux, const double *uy)
{
    //--------------------------------
    //   Create a WINDOW using GLFW
    //--------------------------------

    GLFWwindow *window;

    // initialize the library
    if(!glfwInit())
        return -1;

    // window size for displaying graphics
    int WIDTH  = 800;
    int HEIGHT = 800;

    // set the window's display mode
    window = glfwCreateWindow(WIDTH, HEIGHT, "Flow inside a square cavity", NULL, NULL);
    if(!window) 
    {
        glfwTerminate();
	return -1;
    }

    // make the windows context current
    glfwMakeContextCurrent(window);

    //---------------------------------------
    // Loop until the user closes the window
    //---------------------------------------

    while(!glfwWindowShouldClose(window))
    {
        // render function
        display(window, WIDTH, HEIGHT, ux, uy);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
*/

// D2Q9 parameters 

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

    // define opposite (anti) aections (useful for implementing bounce back)

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

// initialize values for aection vectors, density, velocity and distribution functions on the GPU

void initialize(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, 
                double *ex, double *ey, int *oppos, double *wt,
                double *rho, double *ux, double *uy, double* sigma, 
                double *f, double *feq, double *f_new)
{
    // loop over all voxels
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {

            // natural index for location (i,j)

            int index = i*N+j;  // column-ordering

            // initialize density and velocity fields inside the cavity

              rho[index] = DENSITY;   // density
               ux[index] = 0.0;       // x-component of velocity
               uy[index] = 0.0;       // x-component of velocity
            sigma[index] = 0.0;       // rate-of-strain field

            // specify boundary condition for the moving lid

            if(j==N-1) ux[index] = LID_VELOCITY;

            // assign initial values for distribution functions
            // along various aections using equilibriu, functions

            for(int a=0;a<Q;a++) {
        
                int index_f = a + index*Q;

                double edotu = ex[a]*ux[index] + ey[a]*uy[index];
                double udotu = ux[index]*ux[index] + uy[index]*uy[index];

                feq[index_f]   = rho[index] * wt[a] * (1.0 + 3.0*edotu + 4.5*edotu*edotu - 1.5*udotu);
                f[index_f]     = feq[index_f];
                f_new[index_f] = feq[index_f];

            }

        }
    }
}

// this function updates the values of the distribution functions at all points along all directions
// carries out one lattice time-step (streaming + collision) in the algorithm

void collideAndStream(// READ-ONLY parameters (used by this function but not changed)
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

int main(int argc, char* argv[])
{
    //--------------------------------
    //   Create a WINDOW using GLFW
    //--------------------------------

    GLFWwindow *window;

    // initialize the library
    if(!glfwInit())
        return -1;

    // window size for displaying graphics
    int WIDTH  = 800;
    int HEIGHT = 800;

    // set the window's display mode
    window = glfwCreateWindow(WIDTH, HEIGHT, "Flow inside a square cavity", NULL, NULL);
    if(!window) 
    {
        glfwTerminate();
	return -1;
    }

    // make the windows context current
    glfwMakeContextCurrent(window);

    //---------------------------------------
    // Loop until the user closes the window
    //---------------------------------------

    while(!glfwWindowShouldClose(window))
    {
        // allocate memory

        // distribution functions
        double *f = new double[N*N*Q];
        double *feq = new double[N*N*Q];
        double *f_new = new double[N*N*Q];

        // density and velocity
        double *rho = new double[N*N];
        double *ux = new double[N*N];
        double *uy = new double[N*N];

        // rate-of-strain
        double *sigma = new double[N*N];

        // D3Q9 parameters
        double *ex = new double[Q];
        double *ey = new double[Q];
        int *oppos = new int[Q];
        double *wt = new double[Q];

        // fill D3Q9 parameters in constant memory on the GPU
        D3Q9(ex, ey, oppos, wt);

        // launch GPU kernel to initialize all fields
        initialize(N, Q, DENSITY, LID_VELOCITY, ex, ey, oppos, wt, rho, ux, uy, sigma, f, feq, f_new);

        // time integration
        int time=0;
        clock_t t0, tN;
	t0 = clock();
        while(time<TIME_STEPS) {

            time++;     // lattice time


            collideAndStream(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER, ex, ey, oppos, wt, rho, ux, uy, sigma, f, feq, f_new);

            // collideAndStream and everythingElse were originally one kernel
            // they were separated out to make all threads synchronize globally
            // before moving on to the next set of calculations

            everythingElse(N, Q, DENSITY, LID_VELOCITY, REYNOLDS_NUMBER, ex, ey, oppos, wt, rho, ux, uy, sigma, f, feq, f_new);

            // on-the-fly OpenGL graphics
            if(time%100 == 0) displaySolution(window, WIDTH, HEIGHT, ux, uy);

            tN = clock() - t0;
	    std::cout << "Lattice time " << time 
                      << " clock ticks " << tN 
                      << " wall clock time " << tN/CLOCKS_PER_SEC 
                      << " lattice time steps per second = " << (float) CLOCKS_PER_SEC * time / (float) tN 
                      << std::endl;
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

    
    return 0;
}
