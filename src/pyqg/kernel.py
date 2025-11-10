import numpy as np

try:
    import pyfftw
except ModuleNotFoundError:
    pyfftw = None

DTYPE_real = np.float64
DTYPE_com = np.complex128


def _make_zeros(shape, dtype):
    if pyfftw is not None:
        return pyfftw.zeros_aligned(shape, dtype=dtype)
    else:
        return np.zeros(shape, dtype=dtype)


def _make_rfftn(in_arr, out_arr, threads):
    if pyfftw is not None:
        return pyfftw.FFTW(
            in_arr, out_arr, threads=threads, direction="FFTW_FORWARD", axes=(-2, -1)
        )
    else:
        return lambda: np.fft.rfftn(in_arr, axes=(-2, -1), out=out_arr)


def _make_irfftn(in_arr, out_arr, threads):
    if pyfftw is not None:
        return pyfftw.FFTW(
            in_arr, out_arr, threads=threads, direction="FFTW_BACKWARD", axes=(-2, -1)
        )
    else:
        return lambda: np.fft.irfftn(in_arr, axes=(-2, -1), out=out_arr)


class PseudoSpectralKernel:
    def __init__(
        self,
        nz: int,
        ny: int,
        nx: int,
        fftw_num_threads: int = 1,
        has_q_param: bool = False,
        has_uv_param: bool = False,
    ):
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.nl = ny
        self.nk = nx // 2 + 1
        self._a = np.zeros((self.nz, self.nz, self.nl, self.nk), DTYPE_com)
        self._kk = np.zeros((self.nk), DTYPE_real)
        self._ik = np.zeros((self.nk), DTYPE_com)
        self._ll = np.zeros((self.nl), DTYPE_real)
        self._il = np.zeros((self.nl), DTYPE_com)
        self._k2l2 = np.zeros((self.nl, self.nk), DTYPE_real)
        self.num_threads = fftw_num_threads

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
        self.dt = 0.0
        self.t = 0.0
        self.tc = 0

        # friction
        self.rek = 0.0

        # the tendency
        self.dqhdt = self._empty_com()
        self.dqhdt_p = self._empty_com()
        self.dqhdt_pp = self._empty_com()

        if pyfftw is not None:
            self._fft_q_to_qh = pyfftw.FFTW(
                self.q, self.qh, threads=self.ntd, direction='FFTW_FORWARD', axes=(-2, -1)
            )
            self._ifft_qh_to_q = pyfftw.FFTW(
                self.qh, self.q, threads=self.ntd, direction='FFTW_BACKWARD', axes=(-2, -1)
            )
            self._ifft_uh_to_u = pyfftw.FFTW(
                self.uh, self.u, threads=self.ntd, direction='FFTW_BACKWARD', axes=(-2, -1)
            )
            self._ifft_vh_to_v = pyfftw.FFTW(
                self.vh, self.v, threads=self.ntd, direction='FFTW_BACKWARD', axes=(-2, -1)
            )
            if has_uv_param:
                self._fft_du_to_duh = pyfftw.FFTW(
                    self.du, self.duh, threads=self.ntd, direction='FFTW_FORWARD', axes=(-2, -1)
                )
                self._fft_dv_to_dvh = pyfftw.FFTW(
                    self.dv, self.dvh, threads=self.ntd, direction='FFTW_FORWARD', axes=(-2, -1)
                )
            if has_q_param:
                self._fft_dq_to_dqh = pyfftw.FFTW(
                    self.dq, self.dqh, threads=self.ntd, direction='FFTW_FORWARD', axes=(-2, -1)
                )
            self._fft_uq_to_uqh = pyfftw.FFTW(
                self.uq, self.uqh, threads=self.ntd, direction='FFTW_FORWARD', axes=(-2,-1)
            )
            self._fft_vq_to_vqh = pyfftw.FFTW(
                self.vq, self.vqh, threads=self.ntd, direction='FFTW_FORWARD', axes=(-2,-1)
            )
            # dummy ffts for diagnostics
            self._dummy_fft = pyfftw.FFTW(
                self._dummy_fft_in, self._dummy_fft_out, threads=self.ntd, direction='FFTW_FORWARD', axes=(-2, -1)
            )
            self._dummy_ifft = pyfftw.FFTW(
                self._dummy_ifft_in, self._dummy_ifft_out, threads=self.ntd, direction='FFTW_BACKWARD', axes=(-2, -1)
            )
        else:
            self._fft_q_to_qh = self._npfft_q_to_qh
            self._ifft_qh_to_q = self._npifft_qh_to_q
            self._ifft_uh_to_u = self._npifft_uh_to_u
            self._ifft_vh_to_v = self._npifft_vh_to_v
            if has_uv_param:
                self._fft_du_to_duh = self._npfft_du_to_duh
                self._fft_dv_to_dvh = self._npfft_dv_to_dvh
            if has_q_param:
                self._fft_dq_to_dqh = self._npfft_dq_to_dqh
            self._fft_uq_to_uqh = self._npfft_uq_to_uqh
            self._fft_vq_to_vqh = self._npfft_vq_to_vqh
            self._dummy_fft = self._npdummy_fft
            self._dummy_ifft = self._npdummy_ifft

    def _npfft_q_to_qh(self):
        np.fft.rfftn(self._q, axes=(-2, -1), out=self._qh)

    def _npifft_qh_to_q(self):
        np.fft.irfftn(self._qh, axes=(-2, -1), out=self._q)

    def _npifft_uh_to_u(self):
        np.fft.irfftn(self.uh, axes=(-2, -1), out=self.u)

    def _npifft_vh_to_v(self):
        np.fft.irfftn(self.vh, axes=(-2, -1), out=self.v)

    def _npfft_du_to_duh(self):
        np.fft.rfftn(self.du, axes=(-2, -1), out=self.duh)

    def _npfft_dv_to_dvh(self):
        np.fft.rfftn(self.dv, axes=(-2, -1), out=self.dvh)

    def _npfft_dq_to_dqh(self):
        np.fft.rfftn(self.dq, axes=(-2, -1), out=self.dqh)

    def _npfft_uq_to_uqh(self):
        np.fft.rfftn(self.uq, axes=(-2, -1), out=self.uqh)

    def _npfft_vq_to_vqh(self):
        np.fft.rfftn(self.vq, axes=(-2, -1), out=self.vqh)

    def _npdummy_fft(self):
        np.fft.rfftn(self._dummy_fft_in, axes=(-2, -1), out=self._dummy_fft_out)

    def _npdummy_ifft(self):
        np.fft.irfftn(self._dummy_ifft_in, axes=(-2, -1), out=self._dummy_ifft_out)

    def _empty_real(self):
        """Allocate a space-grid-sized variable for use with fftw transformations."""
        return _make_zeros(shape=(self.nz, self.ny, self.nx), dtype=DTYPE_real)

    def _empty_com(self):
        """Allocate a Fourier-grid-sized variable for use with fftw transformations."""
        return _make_zeros(shape=(self.nz, self.nl, self.nk), dtype=DTYPE_com)

    def fft(self, v):
        """"Generic FFT function for real grid-sized variables.
        Not used for actual model ffs."""
        # copy input into memory view
        return np.fft.rfftn(v, axes=(-2, -1))

    def ifft(self, v):
        return np.fft.irfftn(v, axes=(-2, -1))

    def _invert(self):
        np.sum(self._a * np.expand_dims(self._qh, 0), axis=1, out=self.ph)
        np.multiply(np.negative(np.expand_dims(self._il, (0, -1))), self.ph, out=self.uh)
        np.multiply(np.expand_dims(self._ik, (0, 1)), self.ph, out=self.vh)
        self._ifft_uh_to_u()
        self._ifft_vh_to_v()

    def _do_advection(self):
        np.multiply((self.u + np.expand_dims(self.Ubg[: self.nz], (-1, -2))), self._q, out=self.uq)
        np.multiply(self.v, self._q, out=self.vq)
        self._fft_uq_to_uqh()
        self._fft_vq_to_vqh()
        # spectral divergence
        self.dqhdt = np.negative(
            np.expand_dims(self._ik, (0, 1)) * self.uqh
            + np.expand_dims(self._il, (0, -1)) * self.vqh
            + np.expand_dims(self._ikQy[: self.nz], 1) * self.ph
        )

    def _do_uv_subgrid_parameterization(self):
        new_du, new_dv = self.uv_parameterization(self)
        np.copyto(self.du, new_du)
        np.copyto(self.dv, new_dv)
        self._fft_du_to_duh()
        self._fft_dv_to_dvh()
        self.dqhdt = (
            self.dqhdt
            + ((-1 * np.expand_dims(self._il, (0, -1))) * self.duh)
            + (np.expand_dims(self._ik, (0, 1)) * self.dvh)
        )

    def _do_q_subgrid_parameterization(self):
        np.copyto(self.dq, self.q_parameterization(self))
        self._fft_dq_to_dqh()
        self.dqhdt = self.dqhdt + self.dqh

    def _do_friction(self):
        if self.rek:
            self.dqhdt[-1] = self.dqhdt[-1] + (self.rek * self._k2l2 * self.ph[-1])

    def _forward_timestep(self):
        dt1, dt2, dt3, next_ablevel = self._ab3_coeffs[self.ablevel]
        self.ablevel = next_ablevel

        qh_new = np.expand_dims(self.filtr, 0) * (
            self._qh +
            dt1 * self.dqhdt +
            dt2 * self.dqhdt_p +
            dt3 * self.dqhdt_pp
        )
        np.copyto(self._qh, qh_new)
        self._ifft_qh_to_q()
        np.copyto(self._qh, qh_new)
        self.dqhdt_pp = self.dqhdt_p
        self.dqhdt_p = self.dqhdt
        self.tc += 1
        self.t += self._dt

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        self._ab3_coeffs = (
            (self._dt, 0.0, 0.0, 1),
            (1.5 * self._dt, -0.5 * self._dt, 0.0, 2),
            ((23 / 12) * self._dt, (-16 / 12) * self._dt, (5 / 12) * self._dt, 2),
        )
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
        np.copyto(self._q, b)
        self._fft_q_to_qh()

    @property
    def qh(self):
        return self._qh

    @qh.setter
    def qh(self, b):
        np.copyto(self._qh, b)
        self._ifft_qh_to_q()
        np.copyto(self._qh, b)

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
