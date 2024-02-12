import numpy as np

DTYPE_real = np.float64
DTYPE_com = np.complex128

class PseudoSpectralKernel:
    def __init__(
        self,
        nz: int,
        ny: int,
        nx: int,
        has_q_param: bool = False,
        has_uv_param: bool = False,
    ):
        self.nz = nz
        self.ny = ny
        self.nx = ny
        self.nl = ny
        self.nk = nx // 2 + 1
        self._a = np.zeros((self.nz, self.nz, self.nl, self.nk), DTYPE_com)
        self._kk = np.zeros((self.nk), DTYPE_real)
        self._ik = np.zeros((self.nk), DTYPE_com)
        self._ll = np.zeros((self.nl), DTYPE_real)
        self._il = np.zeros((self.nl), DTYPE_com)
        self._k2l2 = np.zeros((self.nl, self.nk), DTYPE_real)

        # initialize FFT inputs / outputs as byte aligned by pyfftw
        self._q = self._empty_real()
        self._qh = self._empty_com()
        self.ph = self._empty_com()
        self.u = self._empty_real()
        self.uh = self._empty_com()
        self.v = self._empty_real()
        self.vh = self._empty_com()
        self.uq = self._empty_real()
        self.uqh = self._empty_com()
        self.vq = self._empty_real()
        self.vqh = self._empty_com()

        # variables for subgrid parameterizations
        if has_uv_param:
            self.du = self._empty_real()
            self.dv = self._empty_real()
            self.duh = self._empty_com()
            self.dvh = self._empty_com()

        if has_q_param:
            self.dq = self._empty_real()
            self.dqh = self._empty_com()

        # dummy variables for diagnostic ffts
        self._dummy_fft_in = self._empty_real()
        self._dummy_fft_out = self._empty_com()
        self._dummy_ifft_in = self._empty_com()
        self._dummy_ifft_out = self._empty_real()

        # time stuff
        self._dt = 0.0
        self.t = 0.0
        self.tc = 0
        self.ablevel = 0

        # friction
        self.rek = 0.0

        # the tendency
        self.dqhdt = self._empty_com()
        self.dqhdt_p = self._empty_com()
        self.dqhdt_pp = self._empty_com()

    def fft_q_to_qh(self):
        self._qh = np.fft.rfftn(self._q, axes=(-2, -1))

    def ifft_qh_to_q(self):
        self._q = np.fft.irfftn(self._qh, axes=(-2, -1))

    def ifft_uh_to_u(self):
        self.u = np.fft.irfftn(self.uh, axes=(-2, -1))

    def ifft_vh_to_v(self):
        self.v = np.fft.irfftn(self.vh, axes=(-2, -1))

    def fft_du_to_duh(self):
        self.duh = np.fft.rfftn(self.du, axes=(-2, -1))

    def fft_dv_to_dvh(self):
        self.dvh = np.fft.rfftn(self.dv, axes=(-2, -1))

    def fft_dq_to_dqh(self):
        self.dqh = np.fft.rfftn(self.dq, axes=(-2, -1))

    def fft_uq_to_uqh(self):
        self.uqh = np.fft.rfftn(self.uq, axes=(-2, -1))

    def fft_vq_to_vqh(self):
        self.vqh = np.fft.rfftn(self.vq, axes=(-2, -1))

    def _dummy_fft(self):
        self._dummy_fft_out = np.fft.rfftn(self._dummy_fft_in, axes=(-2, -1))

    def _dummy_ifft(self):
        self._dummy_ifft_out = np.fft.irfftn(self._dummy_ifft_in, axes=(-2, -1))

    def _empty_real(self):
        """Allocate a space-grid-sized variable for use with fftw transformations."""
        return np.zeros((self.nz, self.ny, self.nx), dtype=DTYPE_real)

    def _empty_com(self):
        """Allocate a Fourier-grid-sized variable for use with fftw transformations."""
        return np.zeros((self.nz, self.nl, self.nk), dtype=DTYPE_com)

    def fft(self, v):
        """"Generic FFT function for real grid-sized variables.
        Not used for actual model ffs."""
        # copy input into memory view
        return np.fft.rfftn(v, axes=(-2, -1))

    def ifft(self, v):
        return np.fft.irfftn(v, axes=(-2, -1))

    def _invert(self):
        self.ph = np.sum(self.a * np.expand_dims(self.qh, 0), axis=1)
        self.uh = np.negative(np.expand_dims(self._il, (0, -1))) * self.ph
        self.vh = np.expand_dims(self._ik, (0, 1)) * self.ph
        self.ifft_uh_to_u()
        self.ifft_vh_to_v()

    def _do_advection(self):
        self.uq = (self.u + np.expand_dims(self.Ubg[: self.nz], (-1, -2))) * self.q
        self.vq = self.v * self.q
        self.fft_uq_to_uqh()
        self.fft_vq_to_vqh()
        # spectral divergence
        self.dqhdt = np.negative(
            np.expand_dims(self._ik, (0, 1)) * self.uqh
            + np.expand_dims(self._il, (0, -1)) * self.vqh
            + np.expand_dims(self._ikQy[: self.nz], 1) * self.ph
        )

    def _do_uv_subgrid_parameterization(self):
        self.du, self.dv = self.uv_parameterization(self)
        self.fft_du_to_duh()
        self.fft_dv_to_dvh()
        self.dqhdt = (
            self.dqhdt
            + ((-1 * np.expand_dims(self._il, (0, -1))) * self.duh)
            + (np.expand_dims(self._ik, (0, 1)) * self.dvh)
        )

    def _do_q_subgrid_parameterization(self):
        self.dq = self.q_parameterization(self)
        self.fft_dq_to_dqh()
        self.dqhdt = self.dqhdt + self.dqh

    def _do_friction(self):
        k = self.nz - 1
        if self.rek:
            self.dqhdt[k] = self.dqhdt[k] + (self.rek * self._k2l2 * self.ph[k])

    def _forward_timestep(self):
        if self.ablevel == 0:
            dt1 = self._dt
            dt2 = 0.0
            dt3 = 0.0
            self.ablevel = 1
        elif self.ablevel == 1:
            dt1 = 1.5*self._dt
            dt2 = -0.5*self._dt
            dt3 = 0.0
            self.ablevel = 2
        else:
            dt1 = 23./12.*self._dt
            dt2 = -16./12.*self._dt
            dt3 = 5./12.*self._dt

        qh_new = np.expand_dims(self.filtr, 0) * (
            self.qh +
            dt1 * self.dqhdt +
            dt2 * self.dqhdt_p +
            dt3 * self.dqhdt_pp
        )
        self.qh = qh_new
        self.dqhdt_pp = self.dqhdt_p
        self.dqhdt_p = self.dqhdt
        self.ifft_qh_to_q()
        self.tc += 1
        self.t += self._dt

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        self.ablevel = 0

    @property
    def kk(self):
        return self._kk

    @kk.setter
    def kk(self, k):
        self._kk = k
        self._ik = 1j * self._kk
        self._k2l2 = (np.expand_dims(self._kk, 0) ** 2) + (np.expand_dims(self._ll, -1) ** 2)

    @property
    def ll(self):
        return self._ll

    @ll.setter
    def ll(self, l):
        self._ll = l
        self._il = 1j * self._ll
        self._k2l2 = (np.expand_dims(self._kk, 0) ** 2) + (np.expand_dims(self._ll, -1) ** 2)


    @property
    def Qy(self):
        return self._Qy

    @Qy.setter
    def Qy(self, Qy):
        self._Qy = np.copy(Qy)
        self._ikQy = 1j * (np.expand_dims(self.kk, 0) * np.expand_dims(self._Qy, -1))

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, b):
        self._q = np.copy(b)
        self.fft_q_to_qh()

    @property
    def qh(self):
        return self._qh

    @qh.setter
    def qh(self, b):
        self._qh = np.copy(b)
        self.ifft_qh_to_q()

    @property
    def ufull(self):
        return self.u + np.expand_dims(self.Ubg, (-1, -2))

    @property
    def vfull(self):
        return self.v

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = np.copy(a).astype(DTYPE_com)


def tendency_forward_euler(dt, dqdt):
    return dt * dqdt


def tendency_ab2(dt, dqdt, dqdt_p):
    dt1 = 1.5 * dt
    dt2 = -0.5 * dt
    return dt1 * dqdt + dt2 * dqdt_p


def tendency_ab3(dt, dqdt, dqdt_p, dqdt_pp):
    dt1 = (23 / 12) * dt
    dt2 = (-16 / 12) * dt
    dt3 = (5 / 12) * dt
    return dt1 * dqdt + dt2 * dqdt_p + dt3 * dqdt_pp
