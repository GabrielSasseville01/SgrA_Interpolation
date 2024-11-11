#Copyright <2020> <Gunther Witzel, Max-Planck-Insitut für Radioastronomie, Bonn,
#gwitzel@mpifr-bonn.mpg.de>
#
#Redistribution and use in source and binary forms, with or without modification, are
#permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this list of
#conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list
#of conditions and the following disclaimer in the documentation and/or other materials
#provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors may be
#used to endorse or promote products derived from this software without specific prior
#written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
#THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
#OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
#TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This module contains eight classes (constants, SingleZoneSgrA, ElectronPlasma, SyncSED,
GaussianProcess, LightCurves, SEDMovie, ParticleSystem) to generate synchrotron and
synchrotron self-Compton spectra, as well as model their temporal evolution. It can generate time
series at various wavelengths, visualize the temporal spectral evolution of the source in form of an
animation, and provides an interface to load the posterior of an approximate Bayesian computation.
Examples on how to generate an animation with this code is provided in the Jupyter notebook
"Animation Demo.ipynb". The model and the Bayesian analysis is described in detail in the
 ApJS article "Rapid Variability of Sgr A* across the Electromagnetic Spectrum", Witzel et al. 2021.
"""

import sys
from math import pi, e
from decimal import Decimal
import numpy as np
import mpmath as mp

import scipy as sp
from scipy import interpolate as intpo

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.colors as col
import matplotlib.gridspec as gridspec
from numba import jit

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def rebin(a, shape):
    """
    A simple routine to rebin 1D arrays to sparser sampling.

    Parameters
        ----------
        a : 1D array
            The data to be rebinned.
        shape : int
            The new number of data points.

    Output
        ----------
        array
        	The rebinned array.
    """
    sh = shape,len(a)//shape
    return a.reshape(sh).mean(-1)

def weighted_percentile(data, percents, weights=None):
    """
    A simple routine to calculate percentiles from data samples and corresponding weights.

    Parameters
        ----------
        data : array_like
            The data samples.
        percents : array_like of float
            Specifying the quantile(s) (e.g., 0.05 for the 5% quantile).
        weights : array_like
            The weights for each data sample.
    Output
        ----------
        scalar or ndarray
        	The (weighted) percentile.
    """
    if weights is None:
        return np.percentile(data, 100*percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()
    y=np.interp(percents, p, d)

    return y

def weighted_random_ind(weights):
    """
    Routine to draw indices according to their corresponding weights.

    Parameters
        ----------
        weights : array_like
            Array of weights.
    Output
        ----------
        int
        	A randomly drawn array index.
    """
    r = np.random.uniform(0.0, np.sum(weights))
    w = np.sort(weights)
    ind  = np.argsort(weights)
    s = 0.0
    i = 0
    while s < r:
        s += w[int(i)]
        i += 1
    return ind[int(i-1)]

def stop(optional_comment=''):
    '''
    stops the program
    optional_comment: can be of any type
    '''
    print('***************************************')
    #print    '*** line number = %s' %str(inspect.currentframe().f_lineno)
    print('*** %s' %str(optional_comment))
    print('***************************************')
    sys.exit('********** ** FORCED STOP ** ********** ')

def SSC_Param(_alpha):
    """
	Routine to interpolate the coefficients b and d given in Marscher 1983.

    Parameters
        ----------
        _alpha : float
            Spectral index (F~f^-alpha).
    Output
        ----------
        float,float
        	b and d for given _alpha.
    """
    _alpha_given=[0.25, 0.5, 0.75, 1.0,2.0]
    _b = [1.8, 3.2, 3.6,3.8,2.2]
    _d = [130,43,18,9.1,4.35]
    tck_b = sp.interpolate.splrep(_alpha_given, _b, s=0)
    tck_d = sp.interpolate.splrep(_alpha_given, _d, s=0)
    return sp.interpolate.splev(_alpha, tck_b, der=0),sp.interpolate.splev(_alpha, tck_d, der=0)

class constants:
    """
    Class for keeping track of physical constants and quantities.
    """
    def __init__(self):
        """
        Setting constants and quantities at class initialization.
        """
        #UNIVERSAL CONSTANTS ARE DEFINED HERE

        #TIME REFERENCE
        self.J2000=51544.5 # in MJD

        #UNITS OF DISTANCE
        self.AU2KM=149597870.700
        self.pc2AU=206263.36898396

        #UNITS OF TIME
        self.YR2DAY = 365.25
        self.YR2MIN = 365.25*1440.
        self.YR2SEC=365.25*86400.

        #UNITS OF VELOCITY
        self.AUYR2KMS=self.AU2KM/self.YR2SEC

        #UNITS OF ENERGY
        self.JOULE2GEV = 6241506479.9632
        self.JOULE2KEV = self.JOULE2GEV * 1.e6
        self.PLANCK = 6.626070150e-34 #J*s
        self.solarlum = 3.828e26 #W
        self.k  = 1.38064852e-23 #m^2*kg/s^2/K

        #GM
        self.GMSUN=365.25*365.25*0.2959122082855911e-3 # Kg * AU^3 / yr^2
        self.GMSUN_4pi2=self.GMSUN/(4.*pi*pi)

        #SPEED OF LIGHT
        self.CKMS=299792.458        # speed of light in km/s
        self.CAUYR=self.CKMS/self.AUYR2KMS  # speed of ligh in AU/yr

        #OTHERS
        self.SIGMA2FWHM=2.*np.sqrt(2.*np.log(2.))
        self.RAD2DEG = 180./pi
        self.RAD2AS = self.RAD2DEG * 3600.
        self.AS2MAS=1000.
        self.GAUSS2MICROGAUSS = 1.e6
        self.KM2M = 1.e3
        self.MHZ2GHZ = 1.e-3
        self.GHZ2HZ = 1.e9

        #SYNCHROTRON CONSTANTS
        self.C3INV = 8.352e9 #[yr * uG^2 * GeV]
        self.C1 = 16.08 #[MHz / (uG * GeV^2)]
        self.C2INV = 32.* pi*pi / (9.*np.sqrt(3)) * self.C3INV * self.C1 #[uG *Sr / GeV * MHZ * yr]
        self.C2 = 1/self.C2INV / 3.154e13 #[Gev / uG / Sr]
        self.C3 = 1/self.C3INV
        self.LAMBDA2SRPERMICROGAUSS = np.sqrt(pi)/2. * (1./ self.AS2MAS / self.RAD2AS) \
        								/self.GAUSS2MICROGAUSS**(1./4.)
        #factor in lamda equation
        self.k1 = (self.CKMS * self.KM2M)**(2.) * 1.e-12*16.08**0.5*1.e-3 * 1.e-26 \
        			*self.JOULE2GEV*4./pi * self.AS2MAS**2. * self.RAD2AS**2.

        #DISTANCE OF SGR A* (GRAVITY redhift paper: https://arxiv.org/pdf/1904.05721.pdf)
        self.DISTSGRAPC = 8178. #pc
        self.DISTSGRAAU = self.DISTSGRAPC * self.pc2AU
        self.DISTSGRAKM= self.DISTSGRAAU * self.AU2KM
        self.DISTSGRAM= self.DISTSGRAKM * 1000.

        #2GM/c^2 SGRA*
        self.SCHWRADM = 1.2e10 #[m]
        self.SCHWRADAU = self.SCHWRADM / self.AU2KM / 1000. #[AU]

        #ELECTRON MASS [kg]
        self.me = 9.10938356e-31

        #submm, NIR, and X-ray frequencies [GHz]
        self.freq_2_2_um = 136269.299
        self.freq_3_7_um = 79310.174
        self.freq_4_5_um = 66620.546


		#S0-2 semi-major axis = 0.1226 arcsec

class SingleZoneSgrA:
    """
    Class for defining parameters of a spherical emission zone.

    Methods
        ----------
        ZoneInfo() : Printing out zone parameters.
    """
    def __init__(self, dist, l, const, B=None, size=None, lam=None):
        """
        Parameters
            ----------
            dist : float
                Distance to emission zone [m].
        	l : float
        	    Slab thickness of emission zone [m].
        	const : object
        		Instance of the class "constants".
        Keywords
            ----------
            B : float
                Magnetic flux density [G].
        	size : float
        	    Diameter of emission zone [m].
        	lam : object
        		Parameter "eta" in Witzel et al. 2021, equation 25.
        	Comment: Either lam  or both B and size need to be set. If lam and either B or size are
        	set the remaining third keyword is calculated.
        """
        self.const = const
        # if no keyword is set complain
        if lam is None and (B is None or size is None):
            print("insufficient number of parameters")
        else:
            self.dist = dist
            self.l = l
        if size is not None:
            self.size = size

            #compute angular diameter[mas]
            self.theta = self.size/self.dist * self.const.RAD2AS * self.const.AS2MAS
        if lam is None:
            self.B = B

            #compute lam (eta) according to equation 25
            self.lam = self.theta/(self.B)**(1./4.)
        else:
            if B is not None:
                self.B = B
                if size is None:
                    self.theta = self.lam * (self.B)**(1./4.)
                    self.size = self.theta * self.dist / (self.const.RAD2AS \
                    				* self.const.AS2MAS)
            else:
                if size is not None:
                    self.size = size
                    self.B = (self.theta/self.lam)**4.
                else:
                    self.B = None
                    self.size = None
                    self.theta = None

    def ZoneInfo(self):
        """
		Method for printing out zone parameters.
        """
        print("                   ")
        print("Single zone paramters")
        print("  lambda [mas/G^(1/4)]: ", '%.8E' % self.lam)
        print("           theta [mas]: ", '%.8E' % self.theta)
        print("              size [m]: ", '%.8E' % self.size)
        print("    size along LOS [m]: ", '%.8E' % self.l)
        print("                 B [G]: ", self.B)
        print("          distance [m]: ", '%.8E' % self.dist)

class ElectronPlasma:
    """
    Class for defining parameters of the electron plasma.

        Methods
        ----------
        CriticalFreq(self,E) : Compute the critical frequency from electron energy.
        Cf2E(self,cf, B = None) : Compute the electron energy from the critical frequency.
        CoolingTimeE(self,E, B = None) : Compute the synchrotron cooling time from electron energy.
        CoolingTimeCf(self,cf, B = None) : Compute the synchrotron cooling time from the critical.
        GammaFacE(self,E) : Compute the gamma factor from electron energy.
        GammaFacCf(self,cf) : Compute the gamma factor from the critical frequency.
        PlasmaInfo(self) : Print out plasma parameters.
        PlasmaSpecificInfo(self) : Printing out more plasma parameters.
    """
    def __init__(self, n0, gamma, Emin, Emax, const, Zone):
        """
        Parameters
             ----------
            n0 : float
                Scaling factor of the electron energy distribution [cm^-3 * GeV^(gamma-1)].
        	gamma : float
        	    Power-law index of the electron energy distribution.
        	Emin : object
        		Minimum electron energy [GeV].
        	Emax : float
        		Maximum electron energy [GeV].
        	const : object
        		Instance of the class "constants".
        	Zone : object
        		Instance of the class "SingleZoneSgrA".
        """
        self.const = const
        self.Zone = Zone
        self.n0 = n0 #[cm^-3 * GeV^(gamma-1)]
        self.gamma = gamma # >0
        self.Emin = Emin #[GeV]
        self.Emax = Emax #[GeV]
        self.cfmax = self.CriticalFreq(self.Emax) #[GHz]
        self.cfmin = self.CriticalFreq(self.Emin) #[GHz]
        self.tcoolmax = self.CoolingTimeE(self.Emax) #[min]
        self.tcoolmin = self.CoolingTimeE(self.Emin) #[min]
        self.gammamax = self.GammaFacE(self.Emax)
        self.gammamin = self.GammaFacE(self.Emin)
        if 1.-self.gamma == 0:
            self.Ne = self.n0 * np.log(self.Emax/self.Emin) #[cm^-3]
        else:
            self.Ne = self.n0 / (-self.gamma+1) * (Emax**(-self.gamma+1) \
            			- Emin**(-self.gamma+1)) #[cm^-3]
        if 2.-self.gamma == 0:
            self.Ue = self.n0 * np.log(self.Emax/self.Emin) #[GeV * cm^-3]
        else:
            self.Ue = self.n0 / (-self.gamma+2) * (Emax**(-self.gamma+2) \
            			- Emin**(-self.gamma+2)) #[GeV * cm^-3]

    def CriticalFreq(self,E):
        """
		Method for computing the critical frequency of an electron with energy E [GeV].

        Parameters
             ----------
            E : float
                Electron energy [GeV].
        Output
            ----------
            float
        	    Critical frequency [GHz].
        """
    # [E] = GeV, output in GHz
        return self.const.C1 * self.Zone.B * self.const.GAUSS2MICROGAUSS * E**2. \
        		* self.const.MHZ2GHZ

    def Cf2E(self, cf, B = None):
        """
		Method for computing the electron energy [GeV] from the critical frequency.

        Parameters
             ----------
            cf : float
                Critical frequency [GHz].
        Keywords
             ----------
             B : float
                 Magnetic flux density [G]; locally overwrites Zone.B
        Output
            ----------
            float
        	    Electron energy [GeV].
        """
        # [cf] = GHz, output in GeV
        if B is None:
            return (cf/self.const.MHZ2GHZ/self.const.C1/self.Zone.B \
            		/self.const.GAUSS2MICROGAUSS)**(1./2.)
        return (cf/self.const.MHZ2GHZ/self.const.C1/B \
            		/self.const.GAUSS2MICROGAUSS)**(1./2.)

    def CoolingTimeE(self,E, B = None):
        """
		Method for computing the synchrotron cooling time [min] from the electron energy.

        Parameters
             ----------
            E : float
                Electron energy [GeV].
        Keywords
             ----------
             B : float
                 Magnetic flux density [G]; locally overwrites Zone.B
        Output
            ----------
            float
        	    Cooling time [min].
        """
        # [E] = GeV, output in min
        if B is None:
            return 1./(self.const.C3 * (self.Zone.B*self.const.GAUSS2MICROGAUSS)**2. * E) \
            		* self.const.YR2MIN
        return 1./(self.const.C3 * (B*self.const.GAUSS2MICROGAUSS)**2. * E) \
            		* self.const.YR2MIN

    def CoolingTimeCf(self,cf, B = None):
        """
		Method for computing the synchrotron cooling time [min] from the critical freqeuncy.

        Parameters
             ----------
            cf : float
                Critical frequency [GHz].
        Keywords
             ----------
             B : float
                 Magnetic flux density [G]; locally overwrites Zone.B
        Output
            ----------
            float
        	    Cooling time [min].
        """
        # [cf] = GHz, output in min
        if B is None:
            return 1./(self.const.C3 * (self.Zone.B*self.const.GAUSS2MICROGAUSS)**2. \
            		* self.Cf2E(cf)) * self.const.YR2MIN
        return 1./(self.const.C3 * (B*self.const.GAUSS2MICROGAUSS)**2. \
            		* self.Cf2E(cf, B = B)) * self.const.YR2MIN

    def GammaFacE(self,E):
        """
		Method for computing the gamma factor from the electron energy.

        Parameters
             ----------
            E : float
                Electron energy [GeV].
        Output
            ----------
            float
        	    The electron gamma factor.
        """
        # [E] = GeV
        return E/self.const.JOULE2GEV/self.const.me/(self.const.CKMS*self.const.KM2M)**2.

    def GammaFacCf(self, cf):
        """
		Method for computing the gamma factor from the critical freqeuncy.

        Parameters
             ----------
            cf : float
                Critical frequency [GHz].
        Output
            ----------
            float
        	    The electron gamma factor.
        """
        # [cf] = GHz
        return self.Cf2E(cf)/self.const.JOULE2GEV/self.const.me/(self.const.CKMS \
        		*self.const.KM2M)**2

    def PlasmaInfo(self):
        """
        Method for printing out plasma parameters.
        """
        print("                   ")
        print("Plasma parameters")
        print("number density scale [cm^-3 * GeV^(gamma-1)]: ", '%.8E' % self.n0)
        print("                              Electron index: ", self.gamma)
        print("                                  Emax [GeV]: ", '%.8E' % self.Emax)
        print("                                  Emin [GeV]: ", '%.8E' % self.Emin)
        print("            critical frequency of Emax [GHz]: ", '%.8E' % self.cfmax)
        print("            critical frequency of Emin [GHz]: ", '%.8E' % self.cfmin)
        print("                  cooling time at Emax [min]: ", '%.8E' % self.tcoolmax)
        print("                  cooling time at Emin [min]: ", '%.8E' % self.tcoolmin)
        print("                          Lorentz factor max: ", '%.8E' % self.gammamax)
        print("                          Lorentz factor min: ", '%.8E' % self.gammamin)
        print("       elctron number density (slap) [cm^-3]: ", '%.8E' % self.Ne)
        print("electron energy density (slap) [GeV * cm^-3]: ", '%.8E' % self.Ue)
        print("     elctron number density (sphere) [cm^-3]: ", '%.8E' % (self.Ne*1.5))

    def PlasmaSpecificInfo(self):
        """
		Method for printing out plasma parameters specific to the NIR.
        """
        print("                   ")
        print("Cooling time 2.2 um [min]: ", self.CoolingTimeCf(136269.299))

class SyncSED:
    """
    Class for computing synchrotron and SSC spectra and related quantities (Moffet 1972).

        Methods
        ----------
        BesselIntF(self,x)
        G(self,_gamma, LowerLimit, UpperLimit)
        IntExact(self,_gamma, _Nu)
        IntFast(self,_gamma, _Nu)
        GPrime(self,_gamma, _Nu=None, IntMethode = 'IntExact', dps = 15)
        Emissivity(self,_Nu, fast = False)
        ElecNumDen(self,_S, _nu, _theta, _B, _gamma, _nu0)
        AbsorptionCoeff(self,_Nu, fast = False)
        IntensityN0Btheta(self, _Nu, fast = False, sphere = False)
        FluxesN0Btheta(self, _Nu, fast = False, sphere = False)
        TOFrequency(self)
        OThinFlux(self, _Nu, fast = False)
        OThickFlux(self, _Nu, fast = False)
        CreateSpec(self,x1, x2, num = 300, fast = False, sphere = False)
        CreatSpecExpTO(self,x1, x2, _nu0, num = 300, sphere = False)
        OThinSpec(self, x1, x2, num = 300)
        OThickSpec(self, x1, x2, num = 300)
        BrightnessT(self, nu, fast = False)
        IsoLambda(self, _lambda, x1, x2, num = 300)
        IsoLambdaGrid(self, x1, x2, num = 300
        SmAdiExp(self, _Nu)
        AdiExpDev(self, x1)
        KappaL(self, _Nu, _lambda, _S, _Nu_S, gamma = None)
        EmAbsSol(self, _Nu, _lambda, gamma = None)
        Fluxes_FL(self, _Nu, _lambda, _S, _Nu_S, dps = 50, gamma = None)
        TOFrequency_FL(self, _lambda, _S, _Nu_S, gamma = None)
        TOFlux_FL(self, _lambda, _S, _Nu_S, gamma = None)
        OThinFlux_FL(self, _Nu, _lambda, _S, _Nu_S, gamma = None)
        OThickFlux_FL(self, _Nu, _lambda, _S, _Nu_S, gamma = None)
        CreateSpec_FL(self, x1, x2, _lambda, _S, _Nu_S, num = 300)
        CreatSpecExpTO_FL(self, x1, x2, _lambda, _S, _Nu_S, _nu0, num = 300)
        OThinSpec_FL(self, x1, x2, _lambda, _S, _Nu_S, num = 300)
        OThickSpec_FL(self, x1, x2, _lambda, _S, _Nu_S, num = 300)
        SmAdiExp_FL(self, _Nu, _lambda, _S, _Nu_S, gamma = None)
        AdiExpDev_FL(self, x1, _lambda, _S, _Nu_S, gamma = None)
        SSC(self, _d_alpha, _nu0, _Nu_m, _S_m, _Nu, _theta, gamma = None)
        SSCLimits(self, _b_alpha, _nu0, _Nu_m, _S_m, _theta)
        CreateSpecSSC(self, x1, x2, _b_alpha, _d_alpha, _nu0, _Nu_m, _S_m, _theta, num = 300)
        F2nuLnu(self, _F, _nu
        SyncSEDInfo(self, dps = 15)
        SyncSEDSpecificInfo(self)
        SyncSED(self, x1, x2, _nu0, num, exact = False, plot = True, sphere = False)
        SyncSED_FL(self, x1, x2, _S, _Nu_S, _nu0, _lambda, _theta, num, mode='F', plot = True, \
    				SSC = True, **kwargs)
    """
    def __init__(self, Nu, const, Zone, Plasma):
        """
        Parameters
             ----------
             Nu :  float
                 Frequency at which to evaluate the spectrum [GHz]
             const : object
                 Instance of the class "constants".
             Zone : object
                 Instance of the class "SingleZoneSgrA".
             Plasma : object
                 Instance of the class "ElecronPlasma".
        """
        self.const = const
        self.Zone = Zone
        self.Plasma = Plasma
        self.Nu = Nu #[GHz]
        self.BesselInt = self.GPrime(self.Plasma.gamma, self.Nu) #
        self.BesselIntPlus = self.GPrime(self.Plasma.gamma+1, self.Nu)
        self.BesselIntLimit = self.GPrime(self.Plasma.gamma, self.Nu, 'IntFast')
        self.BesselIntPlusLimit = self.GPrime(self.Plasma.gamma+1, self.Nu, 'IntFast')
        if self.Zone.B is not None:
            self.epsilon = self.Emissivity(self.Nu)
            self.kappa = self.AbsorptionCoeff(self.Nu)
            self.intensity = self.IntensityN0Btheta(self.Nu)
            self.fluxdensity = self.FluxesN0Btheta(self.Nu)

    def BesselIntF(self,x):
        """
		Method for computing the integral of the modified Bessel function K_(5/3), see Moffet 1972,
		Eq. 2.10.

        Parameters
             ----------
             x : float
                 Frequency in units of critical frequency.
        Output
             ----------
             float
            	 Emission spectrum from a single electron.
        """
        return float(mp.mpf(x) \
        		* mp.quad(lambda z: sp.special.kv(5./3.,float(z)), [x, mp.inf]))

    def G(self, _gamma, LowerLimit, UpperLimit):
        """
		Method for computing the contribution of electrons in an ensemble with various critical
		frequencies to the emission at a given frequency nu (Moffet 1972, Eqs. 2.21, 2.26).

        Parameters
             ----------
             _gamma : float
                 Electron spectral index ("p").
             LowerLimit : float
                 nu/nu_min, with nu_min the critical frequency of an electron with energy gamma_min.
             UpperLimit :
                 nu/nu_max, with nu_max the critical frequency of an electron with energy gamma_max.
        Output
             ----------
             float
                 Electron ensemble integral.
        """
        return mp.quad(lambda x: \
        		self.BesselIntF(x)*x**((_gamma-3.)/2.), [LowerLimit, UpperLimit])

    def IntExact(self, _gamma, _Nu):
        """
		Method for evaluating G at LowerLimit = nu/nu_min and UpperLimit = nu/nu_max.

        Parameters
             ----------
             _gamma : float
                 Electron spectral index ("p").
             _Nu : float
                 Observing frequency in GHz.
        Output
             ----------
             float
                 Electron ensemble integral.
        """
        return float(self.G(_gamma, _Nu/self.Plasma.cfmax, _Nu/self.Plasma.cfmin) * 0.5 \
        		* pi**0.5 * sp.special.gamma((_gamma+5.)/4.) \
        		/ sp.special.gamma((_gamma+7.)/4.))

    def IntFast(self, _gamma, _Nu):
        """
		Method for computing G in the limit LowerLimit=inf and UpperLimit=0 (Moffet 1972, Eqs. 2.24,
		2.26).

        Parameters
             ----------
             _gamma : float
                 Electron spectral index ("p").
             _Nu : float
                 Observing frequency in GHz.
        Output
             ----------
             float
                 Electron ensemble integral.
        """
        return float(self.G(_gamma, 0., np.inf) * 0.5 * pi**0.5 \
        		* sp.special.gamma((_gamma+5.)/4.) / sp.special.gamma((_gamma+7.)/4.))

    def GPrime(self, _gamma, _Nu=None, IntMethode = 'IntExact', dps = 15):
        """
		Method for computing the electron ensemble integral with either the precise method IntExact
		or the faster method IntFast.

        Parameters
             ----------
             _gamma : float
                 Electron spectral index ("p").
        Keywords
            ----------
            _Nu : float
                Observing frequency in GHz.
            IntMethode : binary
                Control the integration mode, exact (IntExact) or in the limit (IntFast).
            dps : int
                Precision for mpmath.
        Output
            ----------
            float
                Electron ensemble integral.
        """
        mp.mp.dps = dps
        method = getattr(self, IntMethode, lambda: "Invalid integration mode")
        return method(_gamma, _Nu)

    def Emissivity(self, _Nu, fast = False):
        """
		Method for computing the synchrotron emissivity, according to Moffet 1972, Eq. 2.27, from
		parameters set in Zone and Plasma input objects.

        Parameters
             ----------
            _Nu : float
                Observing frequency in GHz.
        Keywords
             ----------
             fast : binary
                 Keyword to control the integration mode in GPrime.
        Output
             ----------
             float
                 Emissivity in W/m^3/Hz/str.
        """
        # W/m^3/Hz/str
        if fast is False:
            return 0.5 * self.const.C2 * self.Plasma.n0 * (self.Zone.B \
            		*self.const.GAUSS2MICROGAUSS)**((self.Plasma.gamma+1.)/2.) \
            		* (self.const.C1/_Nu*self.const.MHZ2GHZ)**((self.Plasma.gamma-1.)/2.) \
            		* self.GPrime(self.Plasma.gamma, _Nu) * 1.e6 / self.const.JOULE2GEV
        return 0.5 * self.const.C2 * self.Plasma.n0 \
            		* (self.Zone.B*self.const.GAUSS2MICROGAUSS)**((self.Plasma.gamma+1.)/2.) \
            		* (self.const.C1/_Nu*self.const.MHZ2GHZ)**((self.Plasma.gamma-1.)/2.) \
            		* self.BesselIntLimit * 1.e6 / self.const.JOULE2GEV

    def ElecNumDen(self,_S, _nu, _theta, _B, _gamma, _nu0):
        """
		Method for computing the electron number density from input parameters (does not depend on
		Zone or Plasma objects).

        Parameters
             ----------
             _S : float
                 Flux density at frequency _nu.
             _nu : float
                 Frequency for which _S is given in GHz.
             _theta : float
                 Angular source diameter [mas].
             _B : float
                 Magnetic flux density [G].
             _gamma : float
                 Electron spectral index ("p").
             _nu0 : float
                 Critical frequency of electrons with gamma_max.
        Output
             ----------
             float
                 Electron number density in cm^-3.
        """
        n0 = Decimal(_S)/Decimal(_theta * self.const.DISTSGRAM \
        		/ (self.const.RAD2AS * self.const.AS2MAS) * pi/4. \
        		* (_theta / self.const.AS2MAS / self.const.RAD2AS)**2. * 1.e26) \
        		/Decimal(0.5 * self.const.C2 * (_B*self.const.GAUSS2MICROGAUSS)**((_gamma+1.)/2.) \
        		* (self.const.C1/_nu*self.const.MHZ2GHZ)**((_gamma-1.)/2.) \
        		* self.BesselIntLimit * 1.e6 / self.const.JOULE2GEV)
        if 1.-(_gamma) == 0:
            return float(n0 * Decimal(np.log(self.Plasma.Cf2E(_nu0, B = _B) \
            		/self.Plasma.Emin))) #[cm^-3]
        return float(n0 / Decimal((-_gamma+1)) \
            		* Decimal((self.Plasma.Cf2E(_nu0, B = _B)**(-_gamma+1) \
            		- self.Plasma.Emin**(-_gamma+1)))) #[cm^-3]

    def AbsorptionCoeff(self,_Nu, fast = False, sphere = False):
        """
		Method for computing the synchrotron absorption coefficient, according to Moffet 1972,
		Eq. 2.49, from parameters set in Zone and Plasma input objects.

        Parameters
             ----------
            _Nu : float
                Observing frequency in GHz.
        Keywords
             ----------
             fast : binary
                 Keyword to control the integration mode in GPrime.
             sphere : binary
                 Keyword to change to spherical symmetry. Default is slab. Experimental.
        Output
             ----------
             float
                 Absorption coefficient in m^-1.
        """
        # m^-1
        if fast is False:
            return 0.5 * (self.const.CKMS * self.const.KM2M)**2. * self.const.C2 \
            		*self.const.C1**(self.Plasma.gamma/2.) * self.Plasma.n0 \
            		* (self.Zone.B*self.const.GAUSS2MICROGAUSS)**((self.Plasma.gamma+2.)/2.) \
            		* (_Nu/self.const.MHZ2GHZ)**(-(self.Plasma.gamma+4.)/2.) \
            		* (self.Plasma.gamma+2) * self.GPrime(self.Plasma.gamma+1, _Nu) * 1.e-6
        return 0.5 * (self.const.CKMS * self.const.KM2M)**2. \
            		* self.const.C2 *self.const.C1**(self.Plasma.gamma/2.) * self.Plasma.n0 \
            		* (self.Zone.B*self.const.GAUSS2MICROGAUSS)**((self.Plasma.gamma+2.)/2.) \
            		* (_Nu/self.const.MHZ2GHZ)**(-(self.Plasma.gamma+4.)/2.) \
            		* (self.Plasma.gamma+2) * self.BesselIntPlusLimit * 1.e-6

    def IntensityN0Btheta(self, _Nu, fast = False, sphere = False):
        """
		Method for computing the synchrotron intensity, according to Moffet 1972, Eq. 2.32, from
		parameters set in Zone and Plasma input objects.

        Parameters
             ----------
            _Nu : float
                Observing frequency in GHz.
        Keywords
             ----------
             fast : binary
                 Keyword to control the integration mode in GPrime.
             sphere : binary
                 Keyword to change to spherical symmetry. Default is slab. Experimental.
        Output
             ----------
             float
                 Intensity in W/m^2/Hz/str.
        """
        _kappa = Decimal(self.AbsorptionCoeff(_Nu, fast = fast, sphere=False))
        if sphere is False:
            if float(_kappa) > 0.:
                return float(Decimal(self.Emissivity(_Nu, fast = fast)) / _kappa \
                		* (Decimal(1.)-(-_kappa*Decimal(self.Zone.l)).exp()))
            return 0.0

        if float(_kappa) > 0.:
            return float(Decimal(self.Emissivity(_Nu, fast = fast)) / _kappa \
                		* Decimal(3.)/Decimal(4.) * (Decimal(1.) + ((Decimal(2.) \
                		* _kappa*Decimal(self.Zone.l) + Decimal(1.))\
                		*(-_kappa*Decimal(self.Zone.l)).exp() - Decimal(1.))/(Decimal(2.)\
                		*_kappa*Decimal(self.Zone.l)*_kappa*Decimal(self.Zone.l))))
        return 0.0

    def FluxesN0Btheta(self, _Nu, fast = False, sphere=False):
        """
		Method for computing the synchrotron flux density, according to Moffet 1972, Eq. 2.50, from
		parameters set in Zone and Plasma input objects.

        Parameters
             ----------
            _Nu : float
                Observing frequency in GHz.
        Keywords
             ----------
             fast : binary
                 Keyword to control the integration mode in GPrime.
             sphere : binary
                 Keyword to change to spherical symmetry. Default is slab. Experimental.
        Output
             ----------
             float
                 Flux density in Jy.
        """
        return self.IntensityN0Btheta(_Nu, fast = fast, sphere = sphere) * pi/4. \
        		* (self.Zone.theta / self.const.AS2MAS / self.const.RAD2AS)**2. * 1.e26

    def TOFrequency(self):
        """
		Method for computing the self-absorption turnover frequency from parameters set in Zone and
		Plasma input objects.

        Output
            ----------
            float
            	Self-absorption turnover frequency in GHz.
        """
        #in GHz
        return (0.5 * (self.const.CKMS * self.const.KM2M)**2. * self.const.C2 \
            		*self.const.C1**(self.Plasma.gamma/2.) * self.Plasma.n0 \
            		* (self.Zone.B*self.const.GAUSS2MICROGAUSS)**((self.Plasma.gamma+2.)/2.) \
            		* (self.Plasma.gamma+2) * self.BesselIntPlusLimit * 1.e-6 \
            		* self.Zone.l)**(2./(self.Plasma.gamma+4.)) * self.const.MHZ2GHZ

    def OThinFlux(self, _Nu, fast = False):
        """
		Method for computing the synchrotron optically thin flux density, according to Moffet 1972,
		Eqs. 2.33 and 2.50, from parameters set in Zone and Plasma input objects.


        Parameters
             ----------
            _Nu : float
                Observing frequency in GHz.
        Keywords
             ----------
             fast : binary
                 Keyword to control the integration mode in GPrime.
        Output
             ----------
             float
                 Flux density in Jy.
        """
        return self.Emissivity(_Nu, fast = fast) * self.Zone.l * pi/4. * (self.Zone.theta \
        		/ self.const.AS2MAS / self.const.RAD2AS)**2. * 1.e26

    def OThickFlux(self, _Nu, fast = False):
        """
		Method for computing the synchrotron optically thick flux density, according to Moffet 1972,
		Eqs. 2.34 and 2.50, from parameters set in Zone and Plasma input objects.

        Parameters
             ----------
            _Nu : float
                Observing frequency in GHz.
        Keywords
             ----------
             fast : binary
                 Keyword to control the integration mode in GPrime.
        Output
             ----------
             float
                 Flux density in Jy.
        """
        return self.Emissivity(_Nu, fast = fast) / self.AbsorptionCoeff(_Nu, fast = fast) * pi/4. \
        		* (self.Zone.theta / self.const.AS2MAS / self.const.RAD2AS)**2. * 1.e26

    def CreateSpec(self,x1, x2, num = 300, fast = False, sphere = False):
        """
		Method for computing a synchrotron (flux density) spectrum from parameters set in Zone and
		Plasma input objects.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2.
             fast : binary
                 Keyword to control the integration mode in GPrime.
             sphere : binary
                 Keyword to change to spherical symmetry. Default is slab. Experimental.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        _r = np.logspace(x1, x2, num = num)
        _y = np.zeros(num)
        for i in range(num):
            _y[i] = self.FluxesN0Btheta(_r[i], fast = fast, sphere = sphere)
            return _r,_y

    def CreatSpecExpTO(self,x1, x2, _nu0, num = 300, sphere = False):
        """
		Method for computing a synchrotron (flux density) spectrum from parameters set in Zone and
		Plasma input objects with exponential cutoff at frequency _nu0.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
            _nu0 : float
                Exponential cutoff frequency in GHz.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2.
             fast : binary
                 Keyword to control the integration mode in GPrime.
             sphere : binary
                 Keyword to change to spherical symmetry. Default is slab. Experimental.
        Output
             ----------
             array,array,array
                 Array of frequencies in GHz, array of flux densities in Jy, array of exponentially
                 truncated flux densities in Jy.
        """
        _r,_y = self.CreateSpec(x1,x2, num = num, fast = True)
        _yy = np.zeros(num)
        for i in range(num):
            _yy[i] = self.FluxesN0Btheta(_r[i], fast=True, sphere = sphere) \
            			* np.exp(-(_r[i]/_nu0)**(1./2.))
        return _r, _y, _yy

    def OThinSpec(self, x1, x2, num = 300):
        """
		Method for computing an optically thin synchrotron (flux density) spectrum from parameters
		set in Zone and Plasma input objects.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        _r = np.logspace(x1, x2, num = num)
        _y = np.zeros(num)
        for i in range(num):
            _y[i] = self.OThinFlux(_r[i], fast = True)
            return _r,_y

    def OThickSpec(self, x1, x2, num = 300):
        """
		Method for computing an optically thick synchrotron (flux density) spectrum from parameters
		set in Zone and Plasma input objects.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        _r = np.logspace(x1, x2, num = num)
        _y = np.zeros(num)
        for i in range(num):
            _y[i] = self.OThickFlux(_r[i], fast = True)
        return _r,_y

    def BrightnessT(self, nu, fast = False):
        """
		Method for computing the synchrotron brightness temperature at frequency nu.

        Parameters
             ----------
             nu : float
                 Observing frequency in GHz.
        Keywords
             ----------
             fast : binary
                 Keyword to control the integration mode in GPrime.
        Output
            ----------
            float
                Synchrotron brightness temperature in K.
        """
        return self.const.PLANCK * nu*1.e9 / self.const.k / np.log(1. + 2.*self.const.PLANCK \
        		* (nu*1.e9)**3. / ((self.const.CKMS* self.const.KM2M)**2. \
        		* self.IntensityN0Btheta(nu, fast = fast)))

    def IsoLambda(self, _lambda, x1, x2, num = 300):
        """
		Method for computing iso-eta lines in the flux density-frequency plane according to
		Moffet 1972, Eq. 2.54.

        Parameters
             ----------
            lambda : eta as defined in Witzel et al. 2021, equation 25
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        _r = np.logspace(x1, x2, num = num)
        _y = np.zeros(num)
        for i in range(num):
            _y[i] = (self.const.CKMS * self.const.KM2M)**(-2.) *self.const.C1**(-1./2.) \
            		* (_lambda*self.const.LAMBDA2SRPERMICROGAUSS)**2. \
            		* (_r[i]/self.const.MHZ2GHZ)**(5./2.) * self.BesselIntLimit \
            		/ ((self.Plasma.gamma+2.) * self.BesselIntPlusLimit) / self.const.JOULE2GEV \
            		* 1.e12 * 1.e26
        return _r,_y

    def IsoLambdaGrid(self, x1, x2, num = 300):
        """
		Method for computing a grid of iso-eta lines in the flux density-frequency plane according
		to Moffet 1972, Eq. 2.54.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency minimum in GHz.
            x2 : float
                Log_10 of the frequency maximum in GHz.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        lambda_array = np.array([1000000., 100000., 10000., 1000., 100., 10., 0.1, 0.01, 0.001,\
        							 0.0001, 0.00001, 0.000001]) * self.Zone.lam
        _y = np.zeros([num, 12])
        for i in range(12):
            _r, _y[:, i] = self.IsoLambda(lambda_array[i], x1, x2, num = num)
        return _r, _y

    def SmAdiExp(self, _Nu):
        """
		Method for computing the self-absorption turnover flux density in the case of adiabatic
		expansion as it develops from the state given by the Zone and Plasma input objects. The flux
		density is computed as a function of future turnover frequency.

        Parameters
             ----------
            _Nu : float
                Turnover frequency in GHz.
        Output
             ----------
             float
                 Turnover flux density in Jy.
        """
        Nu_m = self.TOFrequency()
        S_m = self.OThinFlux(Nu_m, fast = True)
        return S_m*(Nu_m/_Nu)**(-(7.*self.Plasma.gamma+3.)/(4*self.Plasma.gamma+6))

    def AdiExpDev(self, x1):
        """
		Method for computing the self-absorption turnover trajectories for adiabatic expansion in
		the flux density-frequency plane.

        Parameters
             ----------
             x1 : float
                Log_10 of the frequency minimum in GHz.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        Nu_m = self.TOFrequency()
        _r = np.arange(x1, Nu_m)
        _y = np.zeros(_r.size)
        for i in range(_r.size):
            _y[i] = self.SmAdiExp(_r[i])
        return _r,_y

    # Spectra from lambda and optically thin flux

    def KappaL(self, _Nu, _lambda, _S, _Nu_S, gamma = None):
        """
		Method for computing the product of absorption coefficient and slab thickness from eta
		(_lambda) and optically thin flux density _S at frequency _Nu_S, according to
		Witzel et al. 2021, Eq.45.

        Parameters
             ----------
             _Nu :  float
                Observing frequency in GHz.
             _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
             _S :  float
                Optically thin flux density in Jy.
             _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             float
                 Product of absorption coefficient and slab thickness (dimensionless).
        """
        if gamma is None:
            gamma = self.Plasma.gamma
        _beta = (self.const.CKMS * self.const.KM2M)**(2.) * self.const.C1**(1./2.) \
        		* (_lambda*self.const.LAMBDA2SRPERMICROGAUSS)**(-2.) / self.BesselIntLimit \
        		* ((gamma+2.) * self.BesselIntPlusLimit) * 1.e-12
        return (_Nu/self.const.MHZ2GHZ)**(-(gamma+4.)/2.) * _beta * _S * 1.e-26 \
        		* self.const.JOULE2GEV * (_Nu_S/self.const.MHZ2GHZ)**((gamma-1.)/2.)

    def EmAbsSol(self, _Nu, _lambda, gamma = None):
        """
		Method for computing the combined term of Witzel et al. 2021, Eq. 46 from eta
		(_lambda).

        Parameters
             ----------
             _Nu :  float
                Observing frequency in GHz.
             _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             float
                 Product of emissivity over absorption coefficient and the solid angle in Jy.
        """
        if gamma is None:
            gamma = self.Plasma.gamma
        _beta = (self.const.CKMS * self.const.KM2M)**(2.) * self.const.C1**(1./2.) \
        		* (_lambda*self.const.LAMBDA2SRPERMICROGAUSS)**(-2.) / self.BesselIntLimit \
        		* ((gamma+2.) * self.BesselIntPlusLimit) * 1.e-12
        return (_Nu/self.const.MHZ2GHZ)**(5./2.) / _beta / self.const.JOULE2GEV * 1.e26

    def Fluxes_FL(self, _Nu, _lambda, _S, _Nu_S, dps = 50, gamma = None):
        """
		Method for computing the self-absorbed synchrotron flux density at a frequency _Nu from eta
		(_lambda) and optically thin flux density _S at frequency _Nu_S, according to
		Witzel et al. 2021, Eqs.45 and 46.

        Parameters
             ----------
             _Nu :  float
                Observing frequency in GHz.
             _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
             _S :  float
                Optically thin flux density in Jy.
             _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             dps : int
                 Precision for mpmath.
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             float
                 Synchrotron flux density in Jy.
        """
        mp.mp.dps = dps
        return mp.mpf(self.EmAbsSol(_Nu, _lambda), gamma = gamma) \
        		* (mp.mpf(1.)- mp.exp(-mp.mpf(self.KappaL(_Nu, _lambda, _S, _Nu_S, gamma = gamma))))

    def TOFrequency_FL(self, _lambda, _S, _Nu_S, gamma = None):
        """
		Method for computing the self-absorption turnover frequency from eta
		(_lambda) and optically thin flux density _S at frequency _Nu_S.

        Parameters
             ----------
             _Nu :  float
                Observing frequency in GHz.
             _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
             _S :  float
                Optically thin flux density in Jy.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             float
                 Self-absorption turnover frequency in GHz.
        """
        if gamma is None:
            gamma = self.Plasma.gamma
        _beta = (self.const.CKMS * self.const.KM2M)**(2.) * self.const.C1**(1./2.) \
        		* (_lambda*self.const.LAMBDA2SRPERMICROGAUSS)**(-2.) / self.BesselIntLimit \
        		* ((gamma+2.) * self.BesselIntPlusLimit) * 1.e-12
        return (_beta * _S * 1.e-26 * self.const.JOULE2GEV \
        		* (_Nu_S/self.const.MHZ2GHZ)**((self.Plasma.gamma-1.)/2.) )\
        		**(2./(self.Plasma.gamma+4.)) * self.const.MHZ2GHZ

    def TOFlux_FL(self, _lambda, _S, _Nu_S, gamma = None):
        """
		Method for computing the self-absorption turnover flux density from eta
		(_lambda) and optically thin flux density _S at frequency _Nu_S.

        Parameters
             ----------
             _Nu :  float
                Observing frequency in GHz.
             _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
             _S :  float
                Optically thin flux density in Jy.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             float
                 Self-absorption turnover flux density in Jy.
        """
        if gamma is None:
            gamma = self.Plasma.gamma
        return _S*(self.TOFrequency_FL(_lambda, _S, _Nu_S)/_Nu_S)**(-(gamma-1.)/2.)

    def OThinFlux_FL(self, _Nu, _lambda, _S, _Nu_S, gamma = None):
        """
		Method for computing the optically thin flux density at frequency _Nu from eta
		(_lambda) and optically thin flux density _S at frequency _Nu_S.

        Parameters
             ----------
             _Nu :  float
                Observing frequency in GHz.
             _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
             _S :  float
                Optically thin flux density in Jy.
             _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             float
                 Optically thin flux density in Jy.
        """
        if gamma is None:
            gamma = self.Plasma.gamma
        return self.TOFlux_FL(_lambda, _S, _Nu_S) \
        		* (_Nu/self.TOFrequency_FL(_lambda, _S, _Nu_S))**(-(gamma-1.)/2.)

    def OThickFlux_FL(self, _Nu, _lambda, _S, _Nu_S, gamma = None):
        """
		Method for computing the optically thick flux density at frequency _Nu from eta
		(_lambda) and optically thin flux density _S at frequency _Nu_S.

        Parameters
             ----------
             _Nu :  float
                Observing frequency in GHz.
             _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
             _S :  float
                Optically thin flux density in Jy.
             _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             float
                 Optically thick flux density in Jy.
        """
        return self.TOFlux_FL(_lambda, _S, _Nu_S, gamma = gamma) \
        		*(_Nu/self.TOFrequency_FL(_lambda, _S, _Nu_S, gamma = gamma))**(5./2.)

    def CreateSpec_FL(self, x1, x2, _lambda, _S, _Nu_S, num = 300):
        """
		Method for computing the synchrotron (flux density) spectrum from eta (_lambda) and
		optically thin flux density _S at frequency _Nu_S.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
            _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
            _S :  float
                Optically thin flux density in Jy.
            _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        _r = np.logspace(x1, x2, num = num)
        _y = np.zeros(num)
        for i in range(num):
            _y[i] = self.Fluxes_FL(_r[i], _lambda, _S, _Nu_S)
        return _r,_y

    def CreatSpecExpTO_FL(self, x1, x2, _lambda, _S, _Nu_S, _nu0, num = 300):
        """
		Method for computing the synchrotron (flux density) spectrum from eta (_lambda) and
		optically thin flux density _S at frequency _Nu_S. With exponential cutoff at frequency
		_nu0.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
            _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
            _S :  float
                Optically thin flux density in Jy.
            _Nu_S:  float
                Frequency in GHz for which _S was given.
            _nu0 : float
                Exponential cutoff frequency in GHz.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2
        Output
             ----------
             array,array,array
                 Array of frequencies in GHz, array of flux densities in Jy, array of exponentially
                 truncated flux densities in Jy.
        """
        _r,_y = self.CreateSpec_FL(x1,x2, _lambda, _S, _Nu_S, num = num)
        _yy = np.zeros(num)
        for i in range(num):
            _yy[i] = self.Fluxes_FL(_r[i], _lambda, _S, _Nu_S)* np.exp(-(_r[i]/_nu0)**(1./2.))
        return _r, _y, _yy

    def OThinSpec_FL(self, x1, x2, _lambda, _S, _Nu_S, num = 300):
        """
		Method for computing the optically thin (flux density) spectrum from eta (_lambda) and
		optically thin flux density _S at frequency _Nu_S.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
            _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
            _S :  float
                Optically thin flux density in Jy.
            _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        _r = np.logspace(x1, x2, num = num)
        _y = np.zeros(num)
        for i in range(num):
            _y[i] = self.OThinFlux_FL(_r[i], _lambda, _S, _Nu_S)
        return _r,_y

    def OThickSpec_FL(self, x1, x2, _lambda, _S, _Nu_S, num = 300):
        """
		Method for computing the optically thick (flux density) spectrum from eta (_lambda) and
		optically thin flux density _S at frequency _Nu_S.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
            _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
            _S :  float
                Optically thin flux density in Jy.
            _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             num : int
                 number of frequencies between x1 and x2.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        _r = np.logspace(x1, x2, num = num)
        _y = np.zeros(num)
        for i in range(num):
            _y[i] = self.OThickFlux_FL(_r[i], _lambda, _S, _Nu_S)
        return _r,_y

    def SmAdiExp_FL(self, _Nu, _lambda, _S, _Nu_S, gamma = None):
        """
		Method for computing the self-absorption turnover flux density in the case of adiabatic
		expansion as it develops from the state given by the Zone and Plasma input objects. The flux
		density is computed as a function of future turnover frequency from eta (_lambda) and
		optically thin flux density _S at frequency _Nu_S.

        Parameters
             ----------
            _Nu : float
                Turnover frequency in GHz.
            _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
            _S :  float
                Optically thin flux density in Jy.
            _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             float
                 Turnover flux density in Jy.
        """
        if gamma is None:
            gamma = self.Plasma.gamma
        Nu_m = self.TOFrequency_FL(_lambda, _S, _Nu_S, gamma = gamma)
        S_m = self.TOFlux_FL(_lambda, _S, _Nu_S, gamma = gamma)
        return S_m*(Nu_m/_Nu)**(-(7.*gamma+3.)/(4*gamma+6))

    def AdiExpDev_FL(self, x1, _lambda, _S, _Nu_S, gamma = None):
        """
		Method for computing the self-absorption turnover trajectories for adiabatic expansion in
		the flux density-frequency plane from eta (_lambda) and optically thin flux density _S at
		frequency _Nu_S.

        Parameters
             ----------
             x1 : float
                Log_10 of the frequency minimum in GHz.
             _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
             _S :  float
                Optically thin flux density in Jy.
             _Nu_S:  float
                Frequency in GHz for which _S was given.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        Nu_m = self.TOFrequency_FL(_lambda, _S, _Nu_S)
        _r = np.arange(x1, Nu_m)
        _y = np.zeros(_r.size)
        for i in range(_r.size):
            _y[i] = self.SmAdiExp_FL(_r[i], _lambda, _S, _Nu_S, gamma = gamma)
        return _r,_y

    # X-ray SED

    def SSC(self, _d_alpha, _nu0, _Nu_m, _S_m, _Nu, _theta, gamma = None):
        """
		Method for computing the SSC flux density in Jy according to Marscher 1983, Eq.4.

        Parameters
             ----------
             _d_alpha : float
                 d(alpha) as defined in Marscher 1983.
             _nu0 : float
                 Exponential cutoff frequency in GHz.
             _Nu_m : float
                 Sychrotron self-absorption turnover frequency in GHz.
             _S_m : float
                 Sychrotron self-absorption turnover flux density in Jy.
             _Nu : float
                 Observing frequency in GHz.
             _theta : float
                 Angular source diameter.
        Keywords
             ----------
             gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
        Output
            ----------
            float
                SSC flux density in Jy.
        """
        if gamma is None:
            gamma = self.Plasma.gamma
        _alpha = (gamma - 1.)/2.
        return _d_alpha * np.log(_nu0/_Nu_m) * _theta**(-2. * (2.* _alpha + 3.)) \
        		* _Nu_m**(-(3.*_alpha+5.)) * _S_m**(2.*(_alpha + 2.)) \
        		* (self.const.PLANCK * _Nu * self.const.GHZ2HZ \
        		* self.const.JOULE2KEV)**(-_alpha) * 1.e-6

    def SSCLimits(self, _b_alpha, _nu0, _Nu_m, _S_m, _theta):
        """
		Method for computing the frequency limits for the range where the SSC flux density is a
		power-law, Marscher 1983, Eq.5.

        Parameters
             ----------
             _b_alpha : float
                 b(alpha) as defined in Marscher 1983.
             _nu0 : float
                 Exponential cutoff frequency in GHz.
             _Nu_m : float
                 Sychrotron self-absorption turnover frequency in GHz.
             _S_m : float
                 Sychrotron self-absorption turnover flux density in Jy.
             _theta : float
                 Angular source diameter.
        Output
             ----------
             float, float
                 Lower range limit in GHz, upper range limit in GHz.
        """
        return (5.5e-9 * _Nu_m * self.Plasma.gammamin **2.) / (self.const.PLANCK \
        		* self.const.GHZ2HZ * self.const.JOULE2KEV), (0.2/_b_alpha * _theta**(-4.) \
        		* _nu0**2. * _Nu_m**(-5.) * _S_m**2.) / (self.const.PLANCK * self.const.GHZ2HZ \
        		* self.const.JOULE2KEV)

    def CreateSpecSSC(self, x1, x2, _b_alpha, _d_alpha, _nu0, _Nu_m, _S_m, _theta, num = 300):
        """
		Method for computing the SSC (flux density) spectrum.

        Parameters
            ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
            _b_alpha : float
                b(alpha) as defined in Marscher 1983.
            _d_alpha : float
                d(alpha) as defined in Marscher 1983.
            _nu0 : float
                Exponential cutoff frequency in GHz.
            _Nu_m : float
                Sychrotron self-absorption turnover frequency in GHz.
            _S_m : float
                Sychrotron self-absorption turnover flux density in Jy.
            _theta : float
                Angular source diameter.
        Keywords
             ----------
             num : int
                number of frequencies between x1 and x2.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        _r = np.logspace(x1, x2, num = num)
        _y = np.zeros(num)
        _SSCFrqMin, _SSCFrqMax = self.SSCLimits(_b_alpha, _nu0, _Nu_m, _S_m, _theta)
        for i in range(num):
            _y[i] = self.SSC(_d_alpha, _nu0, _Nu_m, _S_m, _r[i], _theta) \
            * np.exp(-((_r[i])/_SSCFrqMax)**(2.)) * np.exp(-(_SSCFrqMin/(_r[i]))**(1.))
        return _r, _y

    def F2nuLnu(self, _F, _nu):
        """
		Method for computing luminosity nuLnu in solar luminosity from flux density in Jy.

        Parameters
             ----------
             _F : float
                 Flux density in Jy at frequency _nu.
             _nu : float
                 Frequency in GHz.
        Output
             ----------
             float
                Luminosity in units of solar luminosity.
        """
        return _F * 1.e-26 * 4. * pi * self.const.DISTSGRAM**2. / self.const.solarlum * _nu * 1.e9

    def SyncSEDInfo(self, dps = 15):
        """
		Method for printing out basic information on the synchotron SED as set by the Zone and
		Plasma object.

        Keywords
             ----------
             dps : int
                 Precision for mpmath.
        """
        mp.mp.dps = dps
        print("                   ")
        print("  SED parameters")
        print("  observing frequency [GHz]: ", self.Nu)
        print("                     G(" + str(self.Plasma.gamma) + "): ", self.G(self.Plasma.gamma,\
        							 self.Nu/self.Plasma.cfmax, self.Nu/self.Plasma.cfmin))
        print("                Gprime(" + str(self.Plasma.gamma) + "): ", self.BesselInt)
        print("               G_limit(" + str(self.Plasma.gamma) + "): ", \
        						self.G(self.Plasma.gamma, 0., np.inf))
        print("          Gprime_limit(" + str(self.Plasma.gamma) + "): ", self.BesselIntLimit)
        if self.Zone.B is not None:
            print("  Emissivity [W/m^3/Hz/str]: ", '%.8E' % self.epsilon)
            print("    Absorption Coeff [m^-1]: ", '%.8E' % self.kappa)
            print("   Intensity [W/m^2/Hz/str]: ", '%.8E' % self.intensity)
            print("          Flux density [Jy]: ", '%.8E' % self.fluxdensity)
        Nu_m = self.TOFrequency()
        print("  Turn-over frequency [GHz]: " , Nu_m)
        print("Turn-over flux density [Jy]: ", self.FluxesN0Btheta(Nu_m, fast = True))
        print("                   S_m [Jy]: ", self.OThinFlux(Nu_m, fast = True))

    def SyncSEDSpecificInfo(self):
        """
		Method for printing out additional information on the synchotron SED.
        """
        print("          ")
        print("            flux density 1THz [Jy]: ", self.OThinFlux(1000., fast = True))
        print("TO frequency from lambda & S [GHz]: ", self.TOFrequency_FL(self.Zone.lam, \
        		self.OThinFlux(1000., fast = True), 1000.))
        print("      TO flux from lambda & S [Jy]: ", self.TOFlux_FL(self.Zone.lam, \
        		self.OThinFlux(1000., fast = True), 1000.))

    def SyncSED(self, x1, x2, _nu0, num, exact = False, plot = True, sphere = False):
        """
		Method for computing and plotting the SED of a synchrotron source including self-absorption
		and SSC contribution.

        Parameters
             ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
            _nu0 : float
                Exponential cutoff frequency in GHz.
            num : int
                number of frequencies between x1 and x2.
        Keywords
             ----------
            exact : binary
                Keyword to control the integration mode in GPrime.
            plot : binary
                Output SED plot.
            sphere : binary
                Keyword to change to spherical symmetry. Default is slab. Experimental.
        Output
             ----------
             array,array
                 Array of frequencies in GHz, array of flux densities in Jy.
        """
        Nu_m = self.TOFrequency()
        if exact is True:
            r_exact, y_exact = self.CreateSpec(x1,x2, num = num, fast = False, sphere = sphere)
        r,y,yy = self.CreatSpecExpTO(x1,x2, _nu0, num = num, sphere = sphere)
        r_thin, y_thin = self.OThinSpec(x1,x2, num = num)
        r_thick, y_thick = self.OThickSpec(x1,x2, num = num)
        r_IsoLam, y_IsoLam = self.IsoLambdaGrid(x1,x2, num = num)
        r_adi, y_adi = self.AdiExpDev(1.0)
        rrr,yyy = self.CreateSpecSSC(x1, x2, SSC_Param((self.Plasma.gamma-1.)/2.)[0], \
        			SSC_Param((self.Plasma.gamma-1.)/2.)[1], _nu0, Nu_m, \
        			self.OThinFlux(Nu_m, fast = True), self.Zone.theta, num = num)
        if plot is True:
            #plt.plot(r,y, color = 'blue')
            plt.plot(r,yy, color = 'orangered')
            plt.plot(r_thin,y_thin, color = 'blue', alpha = 1.0, linestyle=':')
            plt.plot(r_thick,y_thick, color = 'orangered', alpha = 1.0, linestyle=':')
            for i in range(3,9):
                plt.plot(r_IsoLam,y_IsoLam[:,i], color = 'grey', alpha = 0.5, linestyle=':')
                #plt.text(r_IsoLam[-1],y_IsoLam[-1,i], str(lambda_array[i]))
            plt.plot(r_adi, y_adi, color = "darkgreen", alpha = 1.0, linestyle=':')
            plt.plot(rrr,yyy, color = 'darkblue', alpha=0.5)
            plt.plot(rrr,yy+yyy, color = 'red')
            if exact is True:
                plt.plot(r_exact,y_exact, color = 'green', linestyle=':')
                plt.plot(r, yy/y_exact, color = 'black', linestyle=':', alpha = 0.5)
            plt.xscale("log")
            plt.yscale("log")
            plt.ylabel("flux density in Jy")
            plt.xlabel("frequency in GHz")
            plt.axis([10.**x1, 1.1*10.**x2, 0.000000001, 100.0])
            plt.axvline(x=136269.299, alpha = 0.5, linestyle=':')
            plt.axhline(y=2.3, alpha = 0.5, linestyle=':')
            plt.axvline(x=1000., alpha = 0.5, linestyle=':')
            plt.axhline(y=0.01, alpha = 0.5, linestyle=':')
            plt.axhline(y=2.5e-8, alpha = 0.5, linestyle=':')
            plt.axvline(x=Nu_m, alpha = 0.5, linestyle=':')
            x_range=range(480000000, 2000000000, 1000000)
            plt.fill_between(x_range, 0.000000001, 0.0000017, alpha = 0.4)
            plt.show()
            if exact is True:
                plt.plot(r, yy/y_exact, color = 'black', linestyle=':', alpha = 0.5)
                plt.xscale("log")
                plt.show()
        return rrr, yy+yyy

    def SyncSED_FL(self, x1, x2, _S, _Nu_S, _nu0, _lambda, _theta, num, mode='F', plot = True, \
    				SSC = True, **kwargs):
        """
		Method for computing and plotting the SED of a synchrotron source including self-absorption
		and SSC contribution from eta (_lambda) and optically thin flux density _S at frequency
		_Nu_S.

        Parameters
            ----------
            x1 : float
                Log_10 of the frequency start value in GHz.
            x2 : float
                Log_10 of the frequency end value in GHz.
            _S : float
                Optically thin flux density in Jy.
            _Nu_S: float
                Frequency in GHz for which _S was given.
            _nu0 : float
                Exponential cutoff frequency in GHz.
            _lambda :  float
                eta as defined in Witzel et al. 2021, Eq.25.
            _theta : float
                Angular source diameter.
            num : int
                number of frequencies between x1 and x2.
        Keywords
            ----------
            mode : string
                F : SED in Jy.
                nuLnu : SED in solar luminosities.
            plot : binary
                Output SED plot.
            SSC : binary
                Show SSC component in plot.
        Output
            ----------
            array,array,array,array
                Array of frequencies in GHz, array of flux densities in Jy (synch.+SSC), array of
                flux densities in Jy (synch.), array of flux densities in Jy (SSC).
        """
        r_FL, y_FL, yy_FL = self.CreatSpecExpTO_FL(x1,x2, _lambda, _S, _Nu_S, _nu0, num = num)
        r_thin, y_thin = self.OThinSpec_FL(x1,x2, _lambda, _S, _Nu_S, num = num)
        r_thick, y_thick = self.OThickSpec_FL(x1,x2, _lambda, _S, _Nu_S, num = num)
        r_IsoLam, y_IsoLam = self.IsoLambdaGrid(x1,x2, num = num)
        r_adi, y_adi = self.AdiExpDev_FL(1.0, _lambda, _S, _Nu_S)
        Nu_m = self.TOFrequency_FL(_lambda, _S, _Nu_S)
        S_m = self.TOFlux_FL(_lambda, _S, _Nu_S)
        rrr_FL,yyy_FL = self.CreateSpecSSC(x1, x2, SSC_Param((self.Plasma.gamma-1.)/2.)[0], \
        				SSC_Param((self.Plasma.gamma-1.)/2.)[1], _nu0, Nu_m, S_m,
        				_theta, num = num)
        if plot is True:
            #plt.plot(r_FL,y_FL, color = 'blue')
            x_range=range(480000000, 2000000000, 1000000)
            if mode == "F":
                plt.plot(r_FL,yy_FL, color = 'orangered')
                #plt.plot(r_thin,y_thin, color = 'blue', alpha = 1.0, linestyle=':')
                #plt.plot(r_thick,y_thick, color = 'orangered', alpha = 1.0, linestyle=':')
                #for i in range(3,9):
                #    plt.plot(r_IsoLam,y_IsoLam[:,i], color = 'grey', alpha = 0.5, linestyle=':')
                #plt.plot(r_adi, y_adi, color = "green", alpha = 1.0, linestyle=':')
                if SSC is True:
                    plt.plot(rrr_FL,yyy_FL, color = 'darkblue', alpha=0.5)
                    plt.plot(rrr_FL,yy_FL+yyy_FL, color = 'red')
                else:
                    plt.plot(rrr_FL,yy_FL, color = 'red')
                plt.axis([10.**x1, 1.1*10.**x2, 0.000000001, 100.0])
                #plt.axhline(y=2.3, alpha = 0.5, linestyle=':')
                plt.axhline(y=0.01, alpha = 0.5, linestyle=':')
                #plt.axhline(y=2.5e-8, alpha = 0.5, linestyle=':')
                plt.fill_between(x_range, 2.5e-8, 0.0000017, alpha = 0.4)
                plt.ylabel("flux density in Jy")
            elif mode == "nuLnu":
                plt.plot(r_FL,self.F2nuLnu(yy_FL, r_FL), color = 'orangered')
                #plt.plot(r_thin,self.F2nuLnu(y_thin, r_thin), color = 'blue', alpha = 1.0, \
                #			linestyle=':')
                #plt.plot(r_thick,self.F2nuLnu(y_thick, r_thick), color = 'orangered', alpha = 1.0,\
                #			linestyle=':')
                #for i in range(3,9):
                #plt.plot(r_IsoLam,self.F2nuLnu(y_IsoLam[:,i], r_IsoLam), color = 'grey', \
                #			alpha = 0.5, linestyle=':')
                #plt.plot(r_adi, self.F2nuLnu(y_adi, r_adi), color = "green", alpha = 1.0, \
                #			linestyle=':')
                #plt.plot(rrr_FL,self.F2nuLnu(yyy_FL, rrr_FL), color = 'darkblue', alpha=0.5)
                if SSC is True:
                    plt.plot(rrr_FL,self.F2nuLnu(yy_FL+yyy_FL, rrr_FL), color = 'red')
                else:
                    plt.plot(rrr_FL,self.F2nuLnu(yy_FL, rrr_FL), color = 'red')
                plt.axis([10.**x1, 1.1*10.**x2, 1.e-5, 1.e4])
                plt.ylabel(r'$\nu\rm{L}_{\nu}/ \rm{L}_{\rm{sun}}$')

            for key, value in kwargs.items():
                plt.plot(value[:,0], value[:,1], color = "black", alpha = 0.4)
            plt.axvline(x=136269.299, alpha = 0.5, linestyle=':')
            plt.axvline(x=1000., alpha = 0.5, linestyle=':')
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("frequency in GHz")
            Nu_m = self.TOFrequency()
            plt.axvline(x=Nu_m, alpha = 0.5, linestyle=':')
            plt.savefig("SED.png", format='png', dpi=300)
            plt.show()
        return rrr_FL,yy_FL+yyy_FL, yy_FL, yyy_FL

class GaussianProcess:
    """
    Class for computing Gaussian random processes from power-law power spectral densities.

        Methods
        ----------
		PowSpec(self, mode)
		RandomGen(self)
		FFTTimeSerie(self, mode)
		Plot(self, x_min, x_max)
    """

    def __init__(self, params, ranClass=None, mode = None):
        """
        Parameters
            ----------
            params : dictionary
                {        "toff": 0.,      				starting time of time series
                      "restime": self.restime, 			observed cadence
                         "samp": 0.1,     				light curve initial sampling
                       "length": 12.0*self.time_int,  	light curve length
                           "a1": 0.0,     				PSD slope for nu<nu_b
                           "a2": self.PSD_slope_fast,   PSD slope for nu_b<nu<nu_b2
                           "a3": self.PSD_slope_fast,   PSD slope for nu_b2<nu
                         "nu_b": self.PSD_break_fast, 	PSD break frequency
                        "nu_b2": 0.05    				PSD break frequency, nu_b2>nu_b
                 }
        Keywords
            ----------
            ranClass : object
                Object with random number attributes, e.g. a instance of GaussianProcess. For
                providing random numbers for computing correlated Gaussian processes.
            mode : string
                None : Gaussian process
                inner_der : derivative of Gaussian process
        """
        self.__dict__.update(params)
        if ranClass is None:
            self.rand1 = None
            self.rand2 = None
            self.rand3 = None
        else:
            self.rand1 = ranClass.rand1
            self.rand2 = ranClass.rand2
            self.rand3 = ranClass.rand3
        self.nnn = int(round(self.length / self.samp)) - (int(round(self.length / self.samp)) % 2)
        self.N = int(self.nnn / 2)
        self.PowSpec(mode)
        self.RandomGen()
        self.FFTTimeSerie(mode)

    def PowSpec(self, mode):
        """
		Method for generating the power spectral density.

        Parameters:
            ----------
            mode : string
                None : Gaussian process
                inner_der : derivative of Gaussian process
        """
        if mode is None:
            self.Max_nu = 1.0 / (self.samp * 2.0)
            self.Min_nu = 1.0 / (self.nnn * self.samp)
            self.freq = np.arange(0, self.N, dtype=np.uint64) * (self.Max_nu - self.Min_nu) \
            			/ (self.N - 1) + self.Min_nu
            self.ind1 = (self.freq < self.nu_b).nonzero()
            self.ind2 = (self.freq >= self.nu_b).nonzero()
            self.ind3 = (self.freq >= self.nu_b2).nonzero()
            self.pow = np.zeros(self.N, dtype=float)
            self.pow[self.ind1] = self.freq[self.ind1] ** (-self.a1) \
            						* self.nu_b ** (self.a1 - self.a2)
            self.pow[self.ind2] = self.freq[self.ind2] ** (-self.a2)
            self.pow[self.ind3] = self.freq[self.ind3] ** (-self.a3) \
            						* self.nu_b2 ** (self.a3 - self.a2)
            self.f = np.arange(0, self.N + 1, dtype=np.double) / (self.nnn * self.samp)
            self.df = self.f[1]-self.f[0]
            self.spec = np.sqrt(intpo.UnivariateSpline(self.freq, self.pow, s=0, k=3)(self.f))
            self.var = np.sqrt(np.sum(4.0*self.spec[1:-2]**2.0) + self.spec[-1]**2.0)
        elif mode == "inner_der":
            self.Max_nu = 1.0 / (self.samp * 2.0)
            self.Min_nu = 1.0 / (self.nnn * self.samp)
            self.freq = np.arange(0, self.N, dtype=np.uint64) * (self.Max_nu - self.Min_nu) \
            			/ (self.N - 1) + self.Min_nu
            self.ind1 = (self.freq < self.nu_b).nonzero()
            self.ind2 = (self.freq >= self.nu_b).nonzero()
            self.ind3 = (self.freq >= self.nu_b2).nonzero()
            self.pow = np.zeros(self.N, dtype=float)
            self.pow[self.ind1] = self.freq[self.ind1] ** (-self.a1) \
            						* self.nu_b ** (self.a1 - self.a2)
            self.pow[self.ind2] = self.freq[self.ind2] ** (-self.a2)
            self.pow[self.ind3] = self.freq[self.ind3] ** (-self.a3) \
            						* self.nu_b2 ** (self.a3 - self.a2)
            self.f = np.arange(0, self.N + 1, dtype=np.double) / (self.nnn * self.samp)
            self.df = self.f[1]-self.f[0]
            self.spec = np.sqrt(intpo.UnivariateSpline(self.freq, self.pow, s=0, k=3)(self.f))
            self.modifier = np.sin(self.f*self.samp*2.0*pi)/self.samp
            self.spec *= self.modifier
            self.var = np.sqrt(np.sum(4.0*self.spec[1:-2]**2.0) + self.spec[-1]**2.0)

    def RandomGen(self):
        """
		Method for generating random numbers.
        """
        if self.rand1 is None:
            self.rand1 = np.random.standard_normal((self.N - 1,))
            self.rand2 = np.random.standard_normal((self.N - 1,))
            self.rand3 = np.random.standard_normal((1,))
        else:
            print('RandomGen: taking provided random numbers')

    def FFTTimeSerie(self, mode):
        """
		Method for generating Gaussian process via FFT from random numbers.

        Parameters:
            ----------
            mode : string
                None : Gaussian process
                inner_der : derivative of Gaussian process
        """
        self.x = np.zeros(2 * self.N, dtype=np.complex128)
        i_train = [i + 1 for i in range(self.N - 1)]
        j_train = [j + self.N + 1 for j in range(self.N - 1)]
        if mode is None:
            for i in i_train:
                self.x[i] = np.complex(self.spec[i] * self.rand1[i-1], self.spec[i] \
                			* self.rand2[i-1])
            self.x[self.N] = np.complex(self.spec[self.N] * self.rand3, 0.0)
            for j in j_train:
                self.x[j] = np.conj(self.x[2 * self.N - j])
            self.x = np.double(np.real(np.fft.fft(self.x)))
            self.x /= self.var
            self.time = np.double(np.arange(0, self.nnn, dtype=np.uint64) * self.samp) + self.toff
            self.x_res = np.interp(self.restime,self.time,self.x)
        else:
            for i in i_train:
                self.x[i] = np.complex(self.spec[i] * self.rand2[i-1], -self.spec[i] \
                			* self.rand1[i-1])
            self.x[self.N] = np.complex(0.0, self.spec[self.N] * self.rand3)
            for j in j_train:
                self.x[j] = np.conj(self.x[2 * self.N - j])
            self.x = np.double(np.real(np.fft.fft(self.x)))
            self.x /= self.var
            self.time = np.double(np.arange(0, self.nnn, dtype=np.uint64) * self.samp) + self.toff
            self.x_res = np.interp(self.restime,self.time[1:-1],self.x[1:-1])

    def Plot(self, x_min, x_max):
        """
        Method for plotting generated time series.

        Parameters
            ----------
            x_min : float
                Start time in min.
            x_max : float
                Stop time in min.
        """
        plt.plot(self.time,self.x, alpha = 0.5, color = 'blue')
        plt.scatter(self.restime,self.x_res, s = 2.0, marker = 'o' ,color = 'darkblue')
        plt.axis([x_min, x_max,-4.0, 4.0])
        plt.show()

class LightCurves:
    """
    Class for computing light curves with via non-linear transformations from Gaussian processes.

        Methods
        ----------
        LognTrans(self)
        PowerLawTrans(self)
        Plot(self, _time, x_min, x_max, plotNoise = False)
        Derivative(self)
        lambda_loop(self, a, _gamma, B0, f0_B, theta0, f0_theta, df = None)
    """

    def __init__(self, params, GaussianPrClass):
        """
        Parameters
            ----------
            params : dictionary
                {     "mode": "logn",            transformation mode (string, logn or powerlaw)
                      "par1": self.mu_fast, 	 transformation parameter (e.g., mu)
                      "par2": self.sig_fast,     transformation parameter (e.g.,sigma)
                    "offset": 0.0,               flux density offset
                       "fac": 1.,                scaling factor
                "noise_mode": "gauss",           noise mode (string, gauss or poisson)
                     "noise": 0.0003             noise parameter
                }
            GaussianPrClass : object
                Instance of class GaussianProcess.
        """
        self.__dict__.update(params)
        self.x = GaussianPrClass.x
        self.time = GaussianPrClass.time
        self.const = constants()
        if self.mode == "logn":
            self.LognTrans()
        if self.mode == "powerlaw":
            self.PowerLawTrans()
        if self.mode == "radmod":
            self.RadMod()
        self.dflux = None
        self.restime = None

    def LognTrans(self):
        """
		Method for generating log-normally distributed flux densities.
        """
        self.flux = self.fac * np.exp(self.x * self.par2 + self.par1) + self.offset
        if self.noise_mode == "gauss":
            self.flux_noise = self.flux + self.noise * np.random.standard_normal((self.flux.size,))
        if self.noise_mode == "poisson":
            self.flux_noise = np.random.poisson((1.0/(3.933*self.flux**(-0.03541) \
            					+ 0.6564*self.flux**(-1.107))) * 60.)/60.

    def PowerLawTrans(self):
        """
		Method for generating power-law distributed flux densities.
        """
        self.flux = sp.special.erf(self.x / np.sqrt(2.))
        self.corr = 1. / ((3000. / (-self.par1)) ** (self.par2 - 1.) - 1.)
        _exp = 1. / (1. - self.par2)
        self.flux = self.fac * (-self.par1) * ((0.5 * (1. + self.flux) + self.corr) \
        			/ (1. + self.corr)) ** _exp + self. offset
        if self.noise_mode == "gauss":
            self.flux_noise = self.flux + self.noise * np.random.standard_normal((self.flux.size,))
        if self.noise_mode == "poisson":
            self.flux_noise = np.random.poisson((1.0/(3.933*self.flux**(-0.03541) \
            					+ 0.6564*self.flux**(-1.107))) * 60.)/60.

    def Plot(self, _time, x_min, x_max, plotNoise = False):
        """
        Method for plotting generated time series.

        Parameters
            ----------
            _time : array
                time array
            x_min : float
                Start time in min.
            x_max : float
                Stop time in min.
        Keywords
            ----------
            plotNoise : binary
        """
        plt.plot(_time,self.flux, alpha = 0.5, color = 'blue')
        plt.axis([x_min,x_max,-0.1 * max(self.flux[(_time > x_min) & (_time < x_max)]), \
        			1.1*max(self.flux[(_time > x_min) & (_time < x_max)])])
        plt.show()
        if plotNoise is True:
            plt.plot(_time,self.flux_noise, alpha = 0.5, color = 'blue')
            plt.axis([x_min,x_max, -0.1 * max(self.flux[(_time > x_min) & (_time < x_max)]), \
            			1.1*max(self.flux[(_time > x_min) & (_time < x_max)])])
            plt.show()

    def Derivative(self):
        """
		Method for computing the derivative of transformed light curve via gradient.
        """
        self.dflux = np.gradient(self.flux, self.time)

    def lambda_loop(self, a, _gamma, B0, f0_B, theta0, f0_theta, df = None):
        """
		Method for computing B and theta values that result in loops of the self-absoption
		turnover in the flux density-frequency plane.

        Parameters
             ----------
             a : float
                 Parameter to control the dependence of the loops shape on the flux density range.
             _gamma : float
                 Electron spectral index ("p"). Overrides gamma set in the Plasma object.
             B0 : float
                 Scaling factor of B values.
             f0_B : float
                 Ancor flux density for scaling factor of B values.
             theta0 : float
                 Scaling factor of theta values.
             f0_theta : float
                 Ancor flux density for scaling factor of theta values.
        Keywords
             ----------
             df : float,
                 Flux density derivative if calculated by other routine.
        Output
             ----------
             float,float,float
                 eta, theta, and B
        """
        _alpha = (2./(2.*_gamma+1.))/2.
        _alpha_2 = np.arcsin(1./(2.*_gamma+1.))/2.
        if df is None:
            phi = np.tanh(a * self.dflux / self.flux)
        else:
            phi = np.tanh(a * df)
        _B = 10.**(B0+np.log10(self.flux/f0_B)*(_alpha*(phi + 1)))
        _theta = 10.**(theta0+np.log10(self.flux/f0_theta)*(-_alpha_2*(phi + 1))) \
        		* (self.const.SCHWRADM / self.const.DISTSGRAM * self.const.RAD2AS \
        		* self.const.AS2MAS)
        _lambda = _theta/_B**(0.25)
        return _lambda, _theta, _B

class DataGenerator:
    """
    Class for generating animations of SEDs and light curves.

        Methods
        ----------
        ModelSetup(self)
        ModelInfo(self)
        MovieSetup(self)
        MovieData(self)
        init(self)
        update(self,frame)
        MovieMake(self, name, start_time, stop_time, dpi)
        PlotData(self, start_time, stop_time, name = None, scale1 = None, scale2 = None, \
    				scale3 = None, scale4 = None, mode = None, submm = None)
    """

    def __init__(self, params, time_int):
        """
        Parameters
            ----------
            params : dictionary
                Model parameters.
            time_int : int
                Length of time series in min.
        """
        self.__dict__.update(params)
        self.time_int = time_int
        self.const = None
        self.Zone = None
        self.Plasma = None
        self.SySED = None
        self.GP_fast = None
        self.GP_slow = None
        self.GP_slow_der = None
        self.LC_fast = None
        self.LC_slow = None
        self.B_ydata = None
        self.theta_ydata = None
        self.Nu_m_ydata = None
        self.rv_ydata = None
        self.NIR_xdata = None
        self.NIR_ydata = None
        self.IR_ydata = None
        self.submm_xdata = None
        self.submm_ydata = None
        self.submm_ydata_230 = None
        self.radio_xdata = None
        self.radio_ydata = None
        self.X_xdata = None
        self.X_ydata = None
        self.ne_ydata = None
        self.tmp_array_NIR = None
        self.tmp_array_X = None
        self.tmp_array_alpha = None
        self.tmp_array_submm = None
        self.tmp_array_alpha_submm = None
        self.c_inj = None
        self.conv = None
        self.B_future = None
        self.B_array = None
        self.theta_future = None
        self.theta_array = None
        self.x_future = None
        self.f_future = None
        self.f_future_slow = None
        self.f_future_37 = None
        self.f_future_45 = None
        self.f_future_345GHz = None
        self.f_future_340GHz = None
        self.f_future_230GHz = None
        self.f_future_217GHz = None
        self.f_future_100GHz = None
        self.f_future_22GHz = None
        self.f_future_43GHz = None
        self.f_future_15GHz = None
        self.f_future_X = None
        self.f_future_X_flux = None
        self.NIR_SSC = None
        self.NIR_SSC_37 = None
        self.NIR_SSC_45 = None
        self.Nu_m_X = None
        self.freq_rv = None
        self.S_m_X = None
        self.ne_future = None
        self.rr = None
        self.yy = None
        self.yyy = None
        self.yyyy = None
        self.lambda_array = None
        self.etheta_array = None
        self.eB_array = None
        self.nu_min = None
        self.f_future_X_rebin = None
        self.trapo = None
        self.secax = None
        self.dpi = None
        self.restime = None
        self.time_xdata = np.arange(int(time_int))

        self.NIR_xdata_unmasked = None
        self.NIR_xdata_masked = None
        self.NIR_ydata_unmasked = None
        self.NIR_ydata_masked = None

        self.IR_xdata_unmasked = None
        self.IR_xdata_masked = None
        self.IR_ydata_unmasked = None
        self.IR_ydata_masked = None

        self.submm_xdata_unmasked = None
        self.submm_xdata_masked = None
        self.submm_ydata_unmasked = None
        self.submm_ydata_masked = None

        self.X_xdata_unmasked = None
        self.X_xdata_masked = None
        self.X_ydata_unmasked = None
        self.X_ydata_masked = None

        self.ground_idx_start = None

        self.data = {
            "X": {
                "xdata_unmasked": None,
                "xdata_masked": None,
                "ydata_unmasked": None,
                "ydata_masked": None
            },
            "NIR": {
                "xdata_unmasked": None,
                "xdata_masked": None,
                "ydata_unmasked": None,
                "ydata_masked": None
            },
            "IR": {
                "xdata_unmasked": None,
                "xdata_masked": None,
                "ydata_unmasked": None,
                "ydata_masked": None
            },
            "submm": {
                "xdata_unmasked": None,
                "xdata_masked": None,
                "ydata_unmasked": None,
                "ydata_masked": None
            },
        }

    def ModelSetup(self):
        """
        Method for setting up input dictionaries, source, SED, and time series objects.
        """
        self.restime = np.array(range(0, int(1.2*self.time_int), 100))*0.1
        params_fast = {        "toff": 0.,      \
                            "restime": self.restime, \
                               "samp": 0.1,     \
                             "length": 12.0*self.time_int,  \
                                 "a1": 0.0,     \
                                 "a2": self.PSD_slope_fast,     \
                                 "a3": self.PSD_slope_fast,     \
                               "nu_b": self.PSD_break_fast, \
                              "nu_b2": 0.05    \
                        }
        params_slow = {        "toff": 0.,     \
                            "restime": self.restime, \
                               "samp": 0.1,     \
                             "length": 12.0*self.time_int,  \
                                 "a1": 0.0,     \
                                 "a2": self.PSD_slope_slow,     \
                                 "a3": self.PSD_slope_slow,     \
                               "nu_b": self.PSD_break_slow, \
                              "nu_b2": 0.05    \
                        }
        params_lc_fast = {     "mode": "logn",               \
                               "par1": self.mu_fast, \
                               "par2": self.sig_fast,                 \
                             "offset": 0.0,                  \
                                "fac": 1.,                   \
                         "noise_mode": "gauss",              \
                              "noise": 0.0003                  \
                            }
        params_lc_slow = {     "mode": "logn",    \
                               "par1": self.mu_slow,      \
                               "par2": self.sig_slow,       \
                             "offset": 0.0,        \
                                "fac": 1.,         \
                         "noise_mode": "gauss",    \
                              "noise": 0.0003       \
                            }

        self.const = constants()
        self.Zone = SingleZoneSgrA(self.const.DISTSGRAM, self.size_0*self.const.SCHWRADM, \
        			self.const, B=self.B_0, size = self.size_0*self.const.SCHWRADM)
        self.Plasma = ElectronPlasma(3187.02439895, self.gamma, 0.01, 1.0, self.const, self.Zone)
        self.SySED = SyncSED(245., self.const, self.Zone, self.Plasma)
        self.GP_fast = GaussianProcess(params_fast)
        self.GP_slow = GaussianProcess(params_slow, ranClass = self.GP_fast)
        self.GP_slow_der = GaussianProcess(params_slow, ranClass = self.GP_fast, mode = "inner_der")
        self.LC_fast = LightCurves(params_lc_fast, self.GP_fast)
        self.LC_slow = LightCurves(params_lc_slow, self.GP_slow)
        self.LC_slow.Derivative()

        self.trapo = 0.0275

        self.B_ydata, self.theta_ydata = [], []
        self.Nu_m_ydata, self.rv_ydata = [], []
        self.NIR_xdata, self.NIR_ydata = [], []
        self.IR_ydata = []
        self.submm_xdata, self.submm_ydata, self.submm_ydata_230 = [], [], []
        self.radio_xdata, self.radio_ydata = [], []
        self.X_xdata, self.X_ydata = [], []
        self.ne_ydata = []

        self.tmp_array_NIR = np.full((25),-10000.0)
        self.tmp_array_X = np.full((25),-10000.0)
        self.tmp_array_alpha = np.full((25),-10000.0)

        self.tmp_array_submm = np.full((25),-10000.0)
        self.tmp_array_alpha_submm = np.full((25),-10000.0)

    def ModelInfo(self):
        """
        Method for printing out basic information on the object set up in ModelSetup.
        """
        self.Zone.ZoneInfo()
        self.Plasma.PlasmaInfo()
        self.Plasma.PlasmaSpecificInfo()
        self.SySED.SyncSEDInfo()
        self.SySED.SyncSEDSpecificInfo()
        r, y = self.SySED.SyncSED(0.0, 12.0, 20000., 500, exact = False)
        rr, yy, yyy, yyyy = self.SySED.SyncSED_FL(0.0, 12.0, \
        					self.SySED.OThinFlux(1000., fast = True), 1000., 20000., \
        					self.Zone.lam, self.Zone.theta, 500)
        self.GP_fast.Plot(0., 4000.)
        self.GP_slow.Plot(0., 4000.)
        self.LC_fast.Plot(self.GP_fast.time, 0., 4000.)
        self.LC_slow.Plot(self.GP_slow.time, 0., 4000.)

    def LightCurveData(self, sampling_rate):
        """
        Method for computing all data for populating the animation graphs.
        """
        w1 = 3.933
        w2 = 0.6564
        z1 = -0.03541
        z2 = -1.107
        self.c_inj = 0.0
        multiplier = int(10 * 1.0/sampling_rate)
        #mag_flux_cons = 2.0

        self.conv = self.rate_conv * self.eff_area

        self.B_future = np.zeros(int(self.time_int))
        self.theta_future = np.zeros(int(self.time_int))
        self.x_future = np.zeros(int(self.time_int))
        self.f_future = np.zeros(int(self.time_int))
        self.f_future_slow = np.zeros(int(self.time_int))
        self.f_future_45 = np.zeros(int(self.time_int))
        self.f_future_345GHz = np.zeros(int(self.time_int))
        self.f_future_340GHz = np.zeros(int(self.time_int))
        self.f_future_230GHz = np.zeros(int(self.time_int))
        self.f_future_X = np.zeros(int(self.time_int))
        self.f_future_X_flux = np.zeros(int(self.time_int))
        self.Nu_m_X = np.zeros(int(self.time_int))
        self.freq_rv = np.zeros(int(self.time_int))
        self.S_m_X = np.zeros(int(self.time_int))
        self.ne_future = np.zeros(int(self.time_int))
        self.rr = np.zeros([int(self.time_int), 500])
        self.yy = np.zeros([int(self.time_int), 500])
        self.yyy = np.zeros([int(self.time_int), 500])
        self.yyyy = np.zeros([int(self.time_int), 500])

        self.lambda_array, self.theta_array, self.B_array = \
        		self.LC_slow.lambda_loop(self.model_gain, self.Plasma.gamma, np.log10(self.B_0), \
        									self.f0_B, np.log10(self.size_0), self.f0_theta, \
        									df = self.GP_slow_der.x)

        bb = self.B_0 * (np.exp(self.mu_slow+(self.sig_slow**2.)/2.)/self.f0_B) \
        		**(np.sin(np.arcsin(2./(2.*self.gamma+1.))/2.))
        self.nu_min = self.const.MHZ2GHZ * self.const.C1 * self.const.YR2MIN**2. \
        			/ (1./self.PSD_break_slow/(e-1))**2. / self.const.C3**2. \
        			/ (bb*self.const.GAUSS2MICROGAUSS)**3.


        s = 0
        for i in range(0,multiplier*int(self.time_int),multiplier):
            self.Nu_m_X[s] = self.SySED.TOFrequency_FL(self.lambda_array[i], self.LC_slow.flux[i], \
            					self.const.freq_2_2_um)
            self.S_m_X[s] = self.SySED.TOFlux_FL(self.lambda_array[i], self.LC_slow.flux[i], \
            					self.const.freq_2_2_um)
            self.freq_rv[s] = self.const.freq_2_2_um*self.LC_fast.flux[i] \
            					+ np.maximum(self.Nu_m_X[s],self.nu_min)
            self.rr[s,:], self.yy[s,:], self.yyy[s,:], self.yyyy[s,:] = \
            		self.SySED.SyncSED_FL(0.0, 12.0, self.LC_slow.flux[i],self.const.freq_2_2_um, \
            								self.freq_rv[s], self.lambda_array[i], \
            								self.theta_array[i], 500, plot = False)
            s += 1



        s = 0
        for i in range(0,multiplier*int(self.time_int),multiplier):
            self.B_future[s] = self.B_array[i]
            self.theta_future[s] = self.theta_array[i]/(self.const.SCHWRADM / self.const.DISTSGRAM \
            						* self.const.RAD2AS * self.const.AS2MAS)
            self.x_future[s] = self.GP_fast.time[i]
            self.f_future_slow[s] = self.LC_slow.flux[i]
            self.f_future[s] = self.SySED.Fluxes_FL(self.const.freq_2_2_um, self.lambda_array[i], \
            					self.LC_slow.flux[i], self.const.freq_2_2_um) \
            					* np.exp(-(self.const.freq_2_2_um/self.freq_rv[s])**(1./2.)) \
            					*self.ampfac
            self.f_future_45[s] = \
            			self.SySED.Fluxes_FL(self.const.freq_4_5_um, self.lambda_array[i], \
            			self.LC_slow.flux[i], self.const.freq_2_2_um) \
            			* np.exp(-(self.const.freq_4_5_um/self.freq_rv[s])**(1./2.))*self.ampfac
            self.f_future_345GHz[s] = \
            			self.SySED.Fluxes_FL(345., self.lambda_array[i], \
            			self.LC_slow.flux[i], self.const.freq_2_2_um) \
            			* np.exp(-(345./self.freq_rv[s])**(1./2.))*self.ampfac
            self.f_future_340GHz[s] = \
            			self.SySED.Fluxes_FL(340., self.lambda_array[i], \
            			self.LC_slow.flux[i], self.const.freq_2_2_um) \
            			* np.exp(-(340./self.freq_rv[s])**(1./2.))*self.ampfac
            self.f_future_230GHz[s] = \
            			self.SySED.Fluxes_FL(230., self.lambda_array[i], \
            			self.LC_slow.flux[i], self.const.freq_2_2_um) \
            			* np.exp(-(230./self.freq_rv[s])**(1./2.))*self.ampfac
            self.f_future_X_flux[s] = \
            			self.SySED.SSC(SSC_Param((self.Plasma.gamma-1.)/2.)[1], \
            			self.freq_rv[s], self.Nu_m_X[s], self.S_m_X[s], 1240000000., \
            			self.theta_array[i])*self.ampfac
            if (self.f_future_X_flux[s] * self.rate_conv + self.X_offset * self.eff_area) < 0.02:
                self.f_future_X[s] = np.random.poisson((self.f_future_X_flux[s] * self.rate_conv \
                					+ self.X_offset * self.eff_area) * 120.) / 120.
            else:
                self.f_future_X[s] = np.random.poisson(1.0/(w1*(self.f_future_X_flux[s] \
                					* self.rate_conv + self.X_offset * self.eff_area)**z1 \
                					+w2*(self.f_future_X_flux[s] * self.rate_conv + self.X_offset \
                					* self.eff_area)**z2) * 120.) / 120.
            s += 1

        self.f_future_X_rebin = np.repeat(rebin(self.f_future_X,int(self.time_int/5)),5)


    def update(self,frame):
        """
        Method for creating animation frames and updating it with data.

        Parameters
            ----------
            frame : int
                Frame number.
        """
        n_points = 1./0.1
        sampling = 1.0
        i = int(frame* n_points * sampling)
        self.NIR_xdata.append(self.x_future[int(frame)])
        self.f = self.f_future[int(frame)] + self.noise_NIR/3000. * np.random.randn() #noise!!
        self.f_45 = self.f_future_45[int(frame)] + self.noise_NIR/3000 * np.random.randn() # Modify noise
        self.NIR_ydata.append(self.f)
        self.IR_ydata.append(self.f_45)

        self.f_X = self.f_future_X_rebin[int(frame)] #+ 0.0003 * np.random.randn()
        self.X_ydata.append(self.f_X)

        #noise
        self.f_230GHz = self.f_future_230GHz[int(frame)] + self.noise_230GHz * np.random.randn()
        self.f_340GHz = self.f_future_340GHz[int(frame)] + self.noise_340GHz * np.random.randn() # Added here
        self.submm_ydata.append(self.f_340GHz)
        self.submm_ydata_230.append(self.f_230GHz) # Added here

        self.Nu_m = self.Nu_m_X[int(frame)]
        self.Nu_m_ydata.append(self.Nu_m)
        self.rv = self.freq_rv[int(frame)]
        self.rv_ydata.append(self.rv)

        self.ne = np.log10(self.ne_future[int(frame)])
        self.ne_ydata.append(self.ne)

        self.B = self.B_future[int(frame)]
        self.B_ydata.append(self.B)
        self.theta = self.theta_future[int(frame)]
        self.theta_ydata.append(self.theta)

        self.tmp_array_X = np.roll(self.tmp_array_X,1)
        self.tmp_array_NIR = np.roll(self.tmp_array_NIR,1)
        self.tmp_array_alpha = np.roll(self.tmp_array_alpha,1)
        self.tmp_array_submm = np.roll(self.tmp_array_submm,1)
        self.tmp_array_alpha_submm = np.roll(self.tmp_array_alpha_submm,1)
        self.tmp_array_X[0] = np.log10(self.f_X+0.001)+0.12*np.random.randn()
        self.tmp_array_NIR[0] = np.log10(self.f_future[int(frame)])
        self.tmp_array_alpha[0] = np.log10(self.f_future_45[int(frame)] \
        						/self.f_future[int(frame)])/np.log10(2.2/4.5)
        self.tmp_array_submm[0] = self.f_future_230GHz[int(frame)]*self.trapo
        self.tmp_array_alpha_submm[0] = np.log10(self.f_future_230GHz[int(frame)] \
        								/self.f_future_345GHz[int(frame)])/np.log10(230./345.)
    
    def CalculateLightCurves(self, start_time, stop_time):
        """
        Method to calculate light curves without generating an animation.

        Parameters
            ----------
            start_time : int
                Starting time in min.
            stop_time : int
                Stopping time in min.

        Returns
            -------
            dict
                A dictionary containing the calculated light curve data.
            """
        if start_time >= stop_time:
            raise ValueError('start time must be before stop time')

        for i in range(int(stop_time)):
            self.update(i)
        
        self.NIR_ydata = np.array(self.NIR_ydata)
        self.IR_ydata = np.array(self.IR_ydata)
        self.submm_ydata = np.array(self.submm_ydata)
        self.X_ydata = np.array(self.X_ydata)

        # return self.time_xdata, self.NIR_ydata, self.IR_ydata, self.submm_ydata, self.X_ydata
    
    def NIR_mask(self):

        # Define the parameters
        total_time = int(self.time_int)  # total number of minutes
        total_observation_window = 4 * 76 + 3 * 40 + np.random.randint(-50, 51)  # Same as sub-mm observation window +/- between 0 and 50 minutes

        # Randomly select a starting point for the window, shifted by ~50 minutes
        if self.ground_idx_start is None:
            # Randomly select a starting point for the window
            max_start_time = total_time - total_observation_window
            self.ground_idx_start = np.random.randint(0, max_start_time + 1)
            random_start_time = self.ground_idx_start
        else:
            random_start_time = self.ground_idx_start + np.random.randint(-50, 51)

        # Ensure the shifted window fits within the total time
        if random_start_time < 0:
            random_start_time = 0
        if random_start_time + total_observation_window > total_time:
            random_start_time = total_time - total_observation_window

        # Generate the time array
        time_array = np.arange(total_time)

        # Initialize lists to hold the indices
        selected_indices = []

        # Generate the indices within the random window
        current_time = random_start_time
        while current_time < random_start_time + total_observation_window:
            observation_period = np.random.randint(1, 11)  # Random observation period between 1 and 10 minutes
            for minute in range(observation_period):
                if current_time + minute < random_start_time + total_observation_window:
                    selected_indices.append(current_time + minute)
            current_time += observation_period + np.random.randint(1, 11)  # Random gap between 1 and 10 minutes

        # Convert selected indices to a numpy array
        selected_indices = np.array(selected_indices)

        # Generate the unmasked indices
        masked_indices = np.setdiff1d(time_array, selected_indices)

        # Here we assume time_xdata is the same as time_array for simplicity
        self.NIR_xdata_unmasked = time_array[selected_indices]
        self.NIR_xdata_masked = time_array[masked_indices]
        self.NIR_ydata_unmasked = self.NIR_ydata[selected_indices]
        self.NIR_ydata_masked = self.NIR_ydata[masked_indices]

        self.data['NIR']['xdata_unmasked'] = self.NIR_xdata_unmasked
        self.data['NIR']['xdata_masked'] = self.NIR_xdata_masked
        self.data['NIR']['ydata_unmasked'] = self.NIR_ydata_unmasked
        self.data['NIR']['ydata_masked'] = self.NIR_ydata_masked

    def IR_mask(self):

        self.data['IR']['xdata_unmasked'] = self.time_xdata
        self.data['IR']['xdata_masked'] = np.empty(0)
        self.data['IR']['ydata_unmasked'] = self.IR_ydata
        self.data['IR']['ydata_masked'] = np.empty(0)

    def X_mask(self):

        self.data['X']['xdata_unmasked'] = self.time_xdata
        self.data['X']['xdata_masked'] = np.empty(0)
        self.data['X']['ydata_unmasked'] = self.X_ydata
        self.data['X']['ydata_masked'] = np.empty(0)

    def submm_mask(self):
        # Define the parameters
        total_time = int(self.time_int)  # total number of minutes
        num_epochs = 4  # number of observing epochs
        epoch_duration = 76  # duration of each observing epoch in minutes
        gap_between_epochs = 40  # gap between observing epochs in minutes
        observation_period = 7  # duration of each observation period in minutes
        gap_within_epoch = 5  # gap within each observing epoch in minutes
        minutes_between_observations = observation_period + gap_within_epoch

        # Calculate the total duration of the observation window
        total_observation_window = num_epochs * epoch_duration + (num_epochs - 1) * gap_between_epochs

        # Ensure the window can fit within the total time
        if total_observation_window > total_time:
            raise ValueError("The total observation window exceeds the available time.")

        # Generate the time array
        time_array = np.arange(total_time)

        if self.ground_idx_start is None:
            # Randomly select a starting point for the window
            max_start_time = total_time - total_observation_window
            self.ground_idx_start = np.random.randint(0, max_start_time + 1)

        # Initialize lists to hold the indices
        selected_indices = []

        # Generate the indices for each observing epoch within the random window
        for epoch in range(num_epochs):
            start_epoch = self.ground_idx_start + epoch * (epoch_duration + gap_between_epochs)
            end_epoch = start_epoch + epoch_duration
            
            # Generate indices within each epoch
            current_time = start_epoch
            while current_time < end_epoch:
                for minute in range(observation_period):
                    if current_time + minute < end_epoch:
                        selected_indices.append(current_time + minute)
                current_time += minutes_between_observations

        # Convert selected indices to a numpy array
        selected_indices = np.array(selected_indices)

        # Generate the unmasked indices
        masked_indices = np.setdiff1d(time_array, selected_indices)

        # Here we assume time_xdata is the same as time_array for simplicity
        self.submm_xdata_unmasked = time_array[selected_indices]
        self.submm_xdata_masked = time_array[masked_indices]
        self.submm_ydata_unmasked = self.submm_ydata[selected_indices]
        self.submm_ydata_masked = self.submm_ydata[masked_indices]

        self.data['submm']['xdata_unmasked'] = self.submm_xdata_unmasked
        self.data['submm']['xdata_masked'] = self.submm_xdata_masked
        self.data['submm']['ydata_unmasked'] = self.submm_ydata_unmasked
        self.data['submm']['ydata_masked'] = self.submm_ydata_masked

    def add_noise(self, percentage_removed):
        # Randomly remove 10% of the unmasked data and transfer to masked data
        # for key in ['NIR', 'submm']:
        for key in self.data.keys():
            x_unmasked = self.data[key]['xdata_unmasked']
            y_unmasked = self.data[key]['ydata_unmasked']
            x_masked = self.data[key]['xdata_masked']
            y_masked = self.data[key]['ydata_masked']

            # Determine the number of points to remove
            num_to_remove = int(percentage_removed * len(x_unmasked))

            # Randomly select indices to remove
            remove_indices = np.random.choice(len(x_unmasked), num_to_remove, replace=False)

            # Get the indices to keep
            keep_indices = np.setdiff1d(np.arange(len(x_unmasked)), remove_indices)

            # Update the unmasked and masked data
            self.data[key]['xdata_unmasked'] = x_unmasked[keep_indices]
            self.data[key]['ydata_unmasked'] = y_unmasked[keep_indices]
            self.data[key]['xdata_masked'] = np.concatenate([x_masked, x_unmasked[remove_indices]])
            self.data[key]['ydata_masked'] = np.concatenate([y_masked, y_unmasked[remove_indices]])

    
    def masks(self, percentage_removed=0.0):

        self.submm_mask()
        self.NIR_mask()
        self.IR_mask()
        self.X_mask()

        if percentage_removed != 0:
            self.add_noise(percentage_removed)

        return self.data

    


class ParticleSystem:
    """
    Class for extracting posterior chains from ABC particle files.

        Methods
        ----------
        ReadParticle(self, file)
        ParamSet(self, VarDict, PickParticle = False, i = None)
    """

    def __init__(self,file, delPoints = 0):
        """
        Parameters
            ----------
            file : string
                Particle file name.
        Keywords
            ----------
            delPoints : int
                Delete particles with the delPoints highest weights.

        """
        self.file = file
        self.n_delete = delPoints
        self.ReadParticle(self.file)
        self.params_model = None
        self.PickParticle = None

    def ReadParticle(self, file):
        """
		Method for reading particle file and deleting particles with spuriously high weights.

        Parameters
            ----------
            file : string
                Particle file name.
        """
        self.particle_file = file
        self.names = np.genfromtxt(self.particle_file, dtype='str', encoding=None)
        self.run_ssc = np.genfromtxt(self.particle_file, dtype='float', encoding=None)
        self.weights = self.run_ssc[1:,0]
        self.weights.astype(float)
        self.data = self.run_ssc[1:,:]
        self.data = np.transpose(self.data.astype(float))

        self.weights_sorted = np.sort(self.weights)[::-1]
        self.ind_sorted = np.argsort(self.weights)[::-1]

        if self.n_delete > 0:
            print("deleting " + str(self.n_delete) + " data points with the following weights: ")
            print(self.weights_sorted[0:self.n_delete])
            self.data = np.delete(self.data, self.ind_sorted[0:self.n_delete], axis=1)
            self.weights_del = np.delete(self.weights, self.ind_sorted[0:self.n_delete])

        self.particle_dict = {}
        for i in range(self.names[0,:].size):
            self.particle_dict.update({self.names[0,i]:self.data[i,:]})

    def ParamSet(self, VarDict, PickParticle = False, i = None):
        """
		Method for generating a parameter dictionary; either computes the median for each parameter,
		picks randomly a particle according to the weights, or picks a particle according to given
		index i.

        Parameters
            ----------
            file : string
                Particle file name.
        Keywords
            ----------
            PickParticles : binary
                False : compute parameters as median from particle file.
                True : pick particle.
            i : int
            	Controls behavior of PickParticle.
            	None : Pick random particle according to weights.
            	int : Pick particle with index i.

        """
        self.params_model = {}
        self.PickParticle = PickParticle
        if self.PickParticle is False:
            for key in VarDict.keys():
                self.params_model.update({key:weighted_percentile(self.particle_dict[VarDict[key]],\
                							 0.5, weights=self.weights_del)})
                print(key + " ","[" + VarDict[key] + "]:  ", self.params_model[key])
                print(weighted_percentile(self.particle_dict[VarDict[key]], 0.16, \
                							 weights=self.weights_del))
                print(weighted_percentile(self.particle_dict[VarDict[key]], 0.84, \
                								weights=self.weights_del))
        else:
            if i is None:
                i  = weighted_random_ind(self.weights_del)
            print("particle used: ",i)
            for key in VarDict.keys():
                self.params_model.update({key:self.particle_dict[VarDict[key]][i]})
                print(key + " ","[" + VarDict[key] + "]:  ", self.params_model[key])
