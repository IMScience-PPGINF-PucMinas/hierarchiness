import numpy as np

class EvalHierarchiness:
    def __init__(self, px, py):
        self.nrows, self.ncols = np.max(px), np.max(py)
        self.mxy = np.array([[np.count_nonzero(np.logical_and(px==i+1, py==j+1)) for j in range(self.ncols)] for i in range(self.nrows)])
        self.nelems = np.sum(self.mxy)
        self.size_x = np.sum(self.mxy, axis=1)
        self.size_y = np.sum(self.mxy, axis=0)
        self.mxy_x = self.mxy / self.size_x.reshape(-1,1)
        self.mxy_y = self.mxy / self.size_y
    #enddef

    def _compute_norm_factor(self, norm):
        if norm == "numreg":
            return np.ones_like(self.size_y) / self.size_y.shape[0]
        elif norm == "global":
            return self.size_y / self.nelems
        elif norm == "regsize":
            return self.size_y
        else:
            return np.ones_like(self.size_y)
        #endif
    #enddef

    def _compute_cover_nucl(self):
        count_cover = np.sum(self.mxy_y > 0, axis=0)
        count_nucl = np.sum(self.mxy_x == 1, axis=0)
        return count_cover, count_nucl
    #enddef

    def count_fm(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover, count_nucl = self._compute_cover_nucl()
        res = norm_factor * np.logical_and(count_cover > 1, count_nucl == count_cover)
        # Normaliza os resultados em porcentagem com base no total global
        res_percent = (res / self.nelems) * 100
        return res_percent
    # enddef

    def count_mi(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover, count_nucl = self._compute_cover_nucl()
        res = norm_factor * np.logical_and(count_cover > 1, np.logical_and(count_nucl > 1, count_nucl < count_cover))
        res_percent = (res / self.nelems) * 100
        return res_percent
    # enddef

    def count_fi(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover, count_nucl = self._compute_cover_nucl()
        res = norm_factor * np.logical_and(count_cover > 1, count_nucl == 1)
        res_percent = (res / self.nelems) * 100
        return res_percent
    # enddef

    def count_inst(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover, count_nucl = self._compute_cover_nucl()
        res = norm_factor * np.logical_and(count_cover > 1, count_nucl == 0)
        res_percent = (res / self.nelems) * 100
        return res_percent
    # enddef

    def count_stab(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover = np.sum(self.mxy_y > 0, axis=0)
        count_nucl = np.sum(self.mxy_x == 1, axis=0)
        res = norm_factor * np.logical_and(count_cover == 1, count_nucl == 1)
        res_percent = (res / self.nelems) * 100
        return res_percent
    # enddef

    def count_fs(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover = np.sum(self.mxy_y > 0, axis=0)
        count_nucl = np.sum(self.mxy_x == 1, axis=0)
        indexes = np.argwhere(self.mxy_y == 1)
        count_cover_inv = np.zeros_like(count_cover)
        count_nucl_inv = np.zeros_like(count_cover)
        for i, j in indexes:
            count_cover_inv[j] += np.count_nonzero(self.mxy_y[i, :] > 0)
            count_nucl_inv[j] += np.count_nonzero(self.mxy_y[i, :] == 1)
        res = norm_factor * np.logical_and(count_cover == 1, count_nucl == 0) * np.logical_and(count_cover_inv > 0, count_nucl_inv == count_cover_inv)
        res_percent = (res / self.nelems) * 100
        return res_percent
    # enddef

    def count_ds(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover = np.sum(self.mxy_y > 0, axis=0)
        count_nucl = np.sum(self.mxy_x == 1, axis=0)
        indexes = np.argwhere(self.mxy_y == 1)
        count_cover_inv = np.zeros_like(count_cover)
        count_nucl_inv = np.zeros_like(count_cover)
        for i, j in indexes:
            count_cover_inv[j] += np.count_nonzero(self.mxy_y[i, :] > 0)
            count_nucl_inv[j] += np.count_nonzero(self.mxy_y[i, :] == 1)
        res = norm_factor * np.logical_and(count_cover == 1, count_nucl == 0) * np.logical_and(count_cover_inv > 1, np.logical_and(count_nucl_inv > 1, count_nucl_inv < count_cover_inv))
        res_percent = (res / self.nelems) * 100
        return res_percent
    # enddef

    def count_fd(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover = np.sum(self.mxy_y > 0, axis=0)
        count_nucl = np.sum(self.mxy_x == 1, axis=0)
        indexes = np.argwhere(self.mxy_y == 1)
        count_cover_inv = np.zeros_like(count_cover)
        count_nucl_inv = np.zeros_like(count_cover)
        for i, j in indexes:
            count_cover_inv[j] += np.count_nonzero(self.mxy_y[i, :] > 0)
            count_nucl_inv[j] += np.count_nonzero(self.mxy_y[i, :] == 1)
        res = norm_factor * np.logical_and(count_cover == 1, count_nucl == 0) * np.logical_and(count_cover_inv > 1, count_nucl_inv == 1)
        res_percent = (res / self.nelems) * 100
        return res_percent
    # enddef


    def _compute_norm_factor_dif(self, norm):
        if norm == "global":
            return np.ones_like(self.size_y) / self.nelems
        elif norm == "local":
            return 1 / self.size_y
        else:
            return np.ones_like(self.size_y)
    #enddef

    def eval_nest(self, norm="none"):
        norm_factor = self._compute_norm_factor_dif(norm)
        tmp = np.array([[self.mxy_x[i, j] == 1 for j in range(self.ncols)] for i in range(self.nrows)])
        res = np.sum(tmp * self.mxy * norm_factor, axis=0)
        return res
    #enddef

    def eval_infl(self, norm="none"):
        norm_factor = self._compute_norm_factor_dif(norm)
        tmp = np.array([[self.mxy_x[i, j] < 1 and self.mxy_y[i, j] < 1 for j in range(self.ncols)] for i in range(self.nrows)])
        res = np.sum(tmp * self.mxy * norm_factor, axis=0)
        return res
    #enddef

    def eval_re(self, norm="none"):
        norm_factor = self._compute_norm_factor_dif(norm)
        tmp = np.array([[1 if self.mxy_x[i, j] <= 0.5 else (1.0 - self.mxy_x[i, j]) / self.mxy_x[i, j] for j in range(self.ncols)] for i in range(self.nrows)])
        res = np.sum(tmp * self.mxy * norm_factor, axis=0)
        return res
    #enddef
#endclass

