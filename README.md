    KNDS_ORBITS_AND_SHADOWS-Python3

This is a package for Python 3.10, loaded with Opencv 4.10 and Imageio 2.36, to draw the (animated) shadow of a Kerr--Newman--(anti) de Sitter (KNdS) black hole, possibly equipped with a thin Keplerian accretion disk, radiating as a blackbody. The shadow can be either drawn from any standard image, or only the accretion disk is drawn on a black background.
This code also allows to draw (massive or null) orbits in a KNdS space-time, using different integration methods of the geodesic equation.

---------------------------------------------------------------------------------------------------

First, install Python 3.10 (along with the libraries numpy, scipy, matplotlib, cmath, os, pickle, warnings) and its packages opencv-python 4.10 (https://pypi.org/project/opencv-python/) and imageio 2.36 (https://pypi.org/project/imageio/). The latter package is used to handle create gif files for shadows of black holes.

Alternatively, the package can be installed using pip, via the command "pip install knds_orbits_and_shadows"; see also https://pypi.org/project/knds-orbits-and-shadows/

Next, put the content of the present folder anywhere and in the examples.py file, change the first line to match the directory of the files (and images!).
Execute the file examples.py; it uses all the functions of the programs, so it should be a good indicator of the sanity of the package. It is divided in three parts: the first one tests the orbit and shadow display, the second one creates a file comet.gif depicting an animated orbit and the third one creates a folder figure_gif containing the file figure.gif; this represents the shadow of an RNdS black hole, with a background celestial sphere that moves diagonally. The full execution takes about one minute on a 12-core 2.60 GHz CPU with 16 Go of RAM.

---------------------------------------------------------------------------------------------------

Description of the main functions of the package:



- First, the file 'auxi.py' contains all the auxiliary functions that are needed for the computations, namely the metric matrices, their derivatives and the Christoffel symbols.
It also contains the library for the Blackbody radiation colors from http://www.vendian.org/mncharity/dir3/blackbody/.



- 'orbit.py' computes the trajectory of a test particle in a KNdS space-time.

The synthax is as follows: [Vecc,HAM,CAR]=orbit(Lambda,Mass,Kerr,Newman,IniConds,Form,Tau,N,Mu,Conserv), where Lambda is the cosmological constant, Mass is the mass of the black hole, Kerr is its Kerr parameter and Newman its charge.
The vector IniConds records the initial datum of the geodesic, in Boyer-Lindquist coordinates and SI units, IniConds=(r0,theta0,phi0,\dot{r}0,\dot{theta}0,\dot{\phi}0).
The variable Form denotes the formulation to take for the computation: a string with value "Polar", "Weierstrass" (for Lambda=0), "Euler-Lagrange", "Carter", "Hamilton", "Symplectic Euler p", "Symplectic Euler q", "Verlet" or "Stormer-Verlet".
The variable Tau denotes the maximal affine parameter at which the trajectory is computed, N is the number of discretization points, Mu is the "massiveness" of the particle: Mu=1 for a massive particle and Mu=0 for a photon.
Finally, the variable Conserv is set to 1 if the user is willing to compute the Hamiltonian and Carter constant at each node of the discretization. Otherwise, the user is invited to set Conserv to 0.

The output is as follows: the vector Vecc contains the position (r,theta,phi) (in SI units) of the trajectory at each node k*Tau/N (0<k<N) of the discretization. 
If Conserv is set to 1, then the vectors HAM and CAR respectively contain the value of the Hamiltonian and Carter constant at each node.



- 'shadow.py' shadows a KNdS black hole, with a standard image (jpeg, png...), or an accretion disk, or both.

The synthax is shadow(Lambda,Mass,Kerr,Newman,Image,Accretion_data), where Lambda is the cosmological constant, M is the mass, Kerr is the Kerr parameter and Newman is the charge.
The variable Image is a string formed with the name (with extension) of the picture to transform (the file should be in the same folder as the functions).
The variable Accretion_data is a list with eight entries.
Its first entry is a non-negative integer: set it to 0 yields the shadow without any accretion disk, set it to 1 gives the picture with the accretion disk and otherwise, the value should be even and only the shadow of the accretion disk is drawn, with a resolution equal to the chosen even integer.
The second entry of the list is the inclination angle from the equatorial plane (so that set this angle to 0 means to shadow the black hole, as seen from the equatorial plane).
The third entry is a string setting the type of radiation required. Set it to " " only computes the effects (graviational, Doppler, both, see below) and yields the shift along the disk, displayed as a shade of colors from red (redshift) to blue (blueshift).
    Set this variable to "Blackbody" computes the temperature as a blackbody radiation. Finally, set it to "Custom" allows to specify the inner and outer temperatures (see below).
The fourth variable is a string too, specifying the various shifts to take into account. Set it to "Gravitation" (resp. to "Doppler") only computes the gravitational (resp. Doppler) shift. To take both effects into account, set this variable to "Doppler+". If " " is chosen, none of these effects is computed.
    If these two last variables are set to " " or any other values than the ones just described for the third and/or fourth variable, the color of the accretion disk is arbitrarily set to [R,G,B]=[255,69,0] and the brightness is computed with a linear scale from the outer to the inner radius.
The fifth variable is a vector with two entries: the respective inner and outer radii of the disk (in terms of the Schwarzschild radius).
The sixth variable is a vector with one or three entries: the first one is the accretion rate and the other ones are (if specified) the respective inner and outer temperature of the disk (in Kelvin). These two values are needed only in the case where the option "Custom" is chosen and are ignored otherwise.
The seventh variable is a non-negative integer: the brightness scaling. If it is set to 0, then the brightness is linearly computed as above. Otherwise, it is computed with Planck's law and rescaled using this value. This is to be adjusted case by case.
Finally, the eight entry is an integer. If it is set to 1, then the color bar containing the brightness temperature of the disk is displayed as a legend. This is done only in the case where the temperature in indeed computed, that is if the third variable is not set to " ".

The output is the computed picture, displayed in a matplotlib figure.



- 'gif.py' consists of four functions 'shadow4gif', 'make_gif', 'DatFile4gif' and 'make_gif_with_DatFile' that are designed to create gif files depicting the shadow of a black hole (without accretion) with a moving celestial sphere.

The first function (shadow4gif) is an non-display analogue of the program shadow described above, simplified by removing the code concerning the accretion disk; this is an auxiliary function. It is called as shadow4gif(Lambda,Mass,Kerr,Newman,Image_matrix,Angle), with Lambda,Mass,Kerr,Newman as above. The variable Image_matrix is a hypermatrix of size (Nx,Ny,3) containing the BGR values of pixels of an image with resolution (Nx,Ny) and the last variable Angle is the desired inclination angle of the shadow, with respect to the equatorial plane.
The main function (make_gif) has synthax make_gif(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Lambda,Mass,Kerr,Newman,Angle), with Lambda,Mass,Kerr,Newman,Angle as before.
The variable Nimages is the number of images for the gif animation.
The variable Name (a string) is the name of the folder that will be created by the program and containing the gif file.
The variable Image is the background image to use for the celestial sphere, as above.
The variable Resol is a list with two integers representing the desired resolution for the gif.
The variable Shifts is a list with three entries: the first two correspond to the respective horizontal and vertical shifts (in number of pixels) defining the corner of the starting image. The third entry is a coefficient defining the frame rate of the animation, in number of pixels (i.e. at each step, the portion of the image is shifted by this number of pixels: the higher this number, the lower the frame rate.)
The variable Direction can take eight values: when set to "h+" (resp. to "h-", "v+", "v-", "d1+", "d1-", "d2+", "d2-") the screen moves horizontally from left to right (resp. horizontally from right to left, vertically from bottom to top, vertically from top to bottom, diagonally from bottom-left to top-right, diagonally from top-right to bottom-left, diagonally from top-left to bottom-right, diagonally from bottom-right to top-left). Please note that it is the screen (celestial sphere) that moves, and not the camera: this is important when the black hole is not spherically symmetric.
The variable FPS defines the number of frames per second for the gif animation.

The output is a gif file, created in the new folder Name_gif.


The other two functions are made to create several gifs out of a single set of data, allowing to call the heavy function shadow only once.
More precisely, the function DatFile4gif is called as DatFile4gif(Resol,Lambda,Mass,Kerr,Newman,Angle), with each variable having the same meaning as above. The program creates the new folder 'dat_files' (if it doesn't exist already) and puts there a .dat file, named file_Resol_Lambda_Mass_Kerr_Newman_Angle.dat. This file contains all the variables needed to create any gif that could be made using a command of the form make_gif(-,-,-,Resol,-,-,-,Lambda,Mass,Kerr,Newman,Angle). Basically, the program stores the hypermatrix obtained with the function shadow4gif, applied to a specific hypermatrix Image_matrix of the appropriate size, encoded as a permutation of its pixels. The same permutation can then be applied to any other image of the same size, without having to call shadow again.
The other function make_gif_with_DatFile has the same synthax and output as make_gif. But instead of calling the program shadow, this function looks for a .dat file with appropriate parameters inside the folder 'dat_files' to render the images. If no such file is found, an error is returned and the user should first use the function DatFile4gif to create it.



Although all the functions are coded to have inputs expressed in SI units, it is possible for the user to change this by resetting the values of the fundamental constants used (the gravitational constant, the velocity of light, the electric permittivity of vacuum and the Stefan-Boltzmann constant, respectively denoted in the code by 'GSI', 'cSI', 'e0' and 'sb') in the very first line of the functions orbit, shadow and shadow4gif.



---------------------------------------------------------------------------------------------------

For more details on the equations and modelization, the reader is refered to the article available at https://iopscience.iop.org/article/10.1088/1361-6382/accbfe.
For any question, suggestion, commentary, remark, the user is invited to contact the author by email at arthur.garnier[at]math[dot]cnrs[dot]fr.
