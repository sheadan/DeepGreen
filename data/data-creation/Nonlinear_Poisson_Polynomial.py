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

Ax_array = np.linspace(0.01, 0.29, 4)
Ay_array = np.linspace(0.01, 0.29, 4)
Bx_array = np.linspace(0.01, 0.25, 5)
By_array = np.linspace(0.01, 0.25, 5)
C_array = np.linspace(-5, 5, 11)

total_examples = (Ax_array.shape[0] * Ay_array.shape[0]
                  + Ax_array.shape[0] * Ay_array.shape[0]
                  * Bx_array.shape[0] * By_array.shape[0]
                  * C_array.shape[0])
u_array = np.zeros((total_examples, nx + 1, ny + 1))
f_array = np.zeros((total_examples, nx + 1, ny + 1))

ex_num = 0

for ix in range(Ax_array.shape[0]):
    print("i =", ix)
    Ax = Ax_array[ix]
    for iy in range(Ay_array.shape[0]):
        Ay = Ay_array[iy]

        # Create mesh and define function space
        mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), nx, ny)

        V = FunctionSpace(mesh, "CG", 1)

        # Define boundary condition
        def boundary(x, on_boundary):
            return on_boundary

        g = Constant(0.0)
        bc = DirichletBC(V, g, boundary)

        # Define variational problem
        u = Function(V)
        v = TestFunction(V)
        f = Expression("Ax*pow(x[0]-c,3)+Ay*pow(x[1]-c,3)",
                       degree=2,
                       Ax=Ax,
                       Ay=Ay,
                       c=np.pi)
        F = dot((1 + u**2) * grad(u), grad(v)) * dx - f * v * dx

        # Compute solution
        solve(F == 0, u, bc, solver_parameters={"newton_solver":
                                                {"relative_tolerance": 1e-6}})

        vertex_values_u = u.compute_vertex_values(mesh)
        solution_array = vertex_values_u.reshape((nx + 1, ny + 1))
        u_array[ex_num, :, :] = solution_array

        vertex_values_f = f.compute_vertex_values(mesh)
        forcing_array = vertex_values_f.reshape((nx + 1, ny + 1))
        f_array[ex_num, :, :] = forcing_array

        ex_num = ex_num + 1

for ix in range(Ax_array.shape[0]):
    print("i =", ix)
    Ax = Ax_array[ix]
    for iy in range(Ay_array.shape[0]):
        Ay = Ay_array[iy]
        for jx in range(Bx_array.shape[0]):
            Bx = Bx_array[jx]
            for jy in range(By_array.shape[0]):
                By = By_array[jy]
                for k in range(C_array.shape[0]):
                    C = C_array[k]

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
                    f = Expression("Ax*pow(x[0]-c,3)+Ay*pow(x[1]-c,3)+Bx*pow(x[0]-c,2)+By*pow(x[1]-c,2)+C",
                                   degree=2, Ax=Ax, Ay=Ay,
                                   Bx=Bx, By=By, C=C, c=np.pi)
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
np.save(prefix + 'Polynomial_us', u_array)
np.save(prefix + 'Polynomial_fs', f_array)
