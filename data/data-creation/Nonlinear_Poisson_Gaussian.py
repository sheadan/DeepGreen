# Solve nonlinear Poisson equation
# - div (1 + u^2) grad u(x, y) = f(x, y)
# With Dirichlet boundary conditions

from dolfin import *
import numpy as np

# Define limits of rectangular domain
xmin = 0
xmax = 2 * np.pi
ymin = xmin
ymax = xmax

# Number of subintervals for discretization
nx = 127
ny = 127

A_array = np.hstack((np.linspace(-25, -5, 5), np.linspace(5, 25, 5)))
Bx_array = np.linspace(0, 2 * np.pi, 7)[1:-1]
By_array = np.linspace(0, 2 * np.pi, 7)[1:-1]
C_array = np.linspace(0.1, 4.9, 25)

total_examples = A_array.shape[0] * Bx_array.shape[0] * By_array.shape[0] * C_array.shape[0]
u_array = np.zeros((total_examples, nx + 1, ny + 1))
f_array = np.zeros((total_examples, nx + 1, ny + 1))

ex_num = 0
for i in range(A_array.shape[0]):
    print("i =", i)
    A = A_array[i]
    for j in range(Bx_array.shape[0]):
        Bx = Bx_array[j]
        for k in range(By_array.shape[0]):
            By = By_array[k]
            for l in range(C_array.shape[0]):
                C = C_array[l]

                # Create mesh and define function space
                mesh = RectangleMesh(Point(xmin, ymin),
                                     Point(xmax, ymax),
                                     nx,
                                     ny)

                V = FunctionSpace(mesh, "CG", 1)

                # Define boundary condition
                def boundary(x, on_boundary):
                    return on_boundary

                g = Constant(0.0)
                bc = DirichletBC(V, g, boundary)

                # Define variational problem
                u = Function(V)
                v = TestFunction(V)
                f = Expression("A*exp(-(pow(x[0]-Bx,2)+pow(x[1]-By,2))/(2*pow(C,2)))",
                               degree=2, A=A, Bx=Bx, By=By, C=C)
                F = dot((1 + u**2) * grad(u), grad(v)) * dx - f * v * dx

                # Compute solution
                solve(F == 0, u, bc,
                      solver_parameters={"newton_solver":
                                         {"relative_tolerance": 1e-6}})

                vertex_values_u = u.compute_vertex_values(mesh)
                solution_array = vertex_values_u.reshape((nx + 1, ny + 1))
                u_array[ex_num, :, :] = solution_array

                vertex_values_f = f.compute_vertex_values(mesh)
                forcing_array = vertex_values_f.reshape((nx + 1, ny + 1))
                f_array[ex_num, :, :] = forcing_array

                ex_num = ex_num + 1

prefix = 'S3-NLP_'
np.save(prefix + 'Gaussian_us', u_array)
np.save(prefix + 'Gaussian_fs', f_array)
