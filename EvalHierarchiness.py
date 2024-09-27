import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import csv
from PIL import Image
from multiprocessing import Pool
import numpy as np

class EvalHierarchiness:
    def __init__(self, px, py):
        self.nrows, self.ncols = np.max(px), np.max(py);
        self.mxy = np.array([[np.count_nonzero(np.logical_and(px==i+1, py==j+1)) for j in range(self.ncols)] for i in range(self.nrows)]);
        self.nelems = np.sum(self.mxy);
        self.size_x = np.sum(self.mxy, axis=1);
        self.size_y = np.sum(self.mxy, axis=0);
        self.mxy_x = self.mxy / self.size_x.reshape(-1,1);
        self.mxy_y = self.mxy / self.size_y;
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
        return res
    # enddef

    def count_mi(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover, count_nucl = self._compute_cover_nucl()
        res = norm_factor * np.logical_and(count_cover > 1, np.logical_and(count_nucl > 1, count_nucl < count_cover))
        return res
    # enddef

    def count_fi(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover, count_nucl = self._compute_cover_nucl()
        res = norm_factor * np.logical_and(count_cover > 1, count_nucl == 1)
        return res
    # enddef

    def count_inst(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover, count_nucl = self._compute_cover_nucl()
        res = norm_factor * np.logical_and(count_cover > 1, count_nucl == 0)
        return res
    # enddef

    def count_stab(self, norm="none"):
        norm_factor = self._compute_norm_factor(norm)
        count_cover = np.sum(self.mxy_y > 0, axis=0)
        count_nucl = np.sum(self.mxy_x == 1, axis=0)
        res = norm_factor * np.logical_and(count_cover == 1, count_nucl == 1)
        return res
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
        return res
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
        return res
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
        return res
    # enddef

    def _compute_norm_factor_dif(self, norm):
        if norm == "global":
            return np.ones_like(self.size_y) / self.nelems
        elif norm == "regional":
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
        # Normal penalization
        # tmp = np.array([[self.mxy_x[i, j] if self.mxy_x[i, j] <= 0.5 else (1.0 - self.mxy_x[i, j]) for j in range(self.ncols)] for i in range(self.nrows)])
        # Harsher penalization
        tmp = np.array([[1 if self.mxy_x[i, j] <= 0.5 else (1.0 - self.mxy_x[i, j]) / self.mxy_x[i, j] for j in range(self.ncols)] for i in range(self.nrows)])
        res = np.sum(tmp * self.mxy * norm_factor, axis=0)
        return res
    #enddef
#endclass

def main(args):
    # Partition A
    pa = np.array([[1,1,2,2,5,5,5,5],
                  [1,1,2,2,5,5,5,5],
                  [3,3,4,4,5,5,5,5],
                  [3,3,4,4,5,5,5,5],
                  [6,6,6,6,7,7,8,8],
                  [6,6,6,6,7,7,8,8],
                  [6,6,6,6,9,9,10,10],
                  [6,6,6,6,9,9,10,10],
                 ]);
     # Partition B
    pb = np.array([[1,1,1,2,5,5,6,6],
                  [1,1,1,2,5,5,6,6],
                  [1,1,1,2,5,5,6,6],
                  [3,3,3,4,8,8,8,8],
                  [7,7,7,7,8,8,8,8],
                  [7,7,7,7,8,8,8,8],
                  [7,7,7,7,8,8,8,8],
                  [7,7,7,7,9,9,10,10],
                 ]);
     # Partition C
    pc = np.array([[1,1,1,1,2,2,2,2],
                  [1,1,1,1,2,2,2,2],
                  [1,1,1,1,2,2,2,2],
                  [1,1,1,1,2,2,2,2],
                  [3,3,3,3,4,4,4,4],
                  [3,3,3,3,4,4,4,4],
                  [3,3,3,3,4,4,4,4],
                  [3,3,3,3,4,4,4,4],
                 ]);

    px = np.array(Image.open(args.x)) if args.x != None else pa;
    py = np.array(Image.open(args.y)) if args.y != None else pb;

    eval_part = EvalHierarchiness(px, py);

    print(50*"-" + "\nHierarchiness Measures\n" + 50*"-");
    if(args.norm == "global" or args.norm == "none" or args.norm == "regional"):
        print("Nestedness(X,Y):", eval_part.eval_nest(args.norm));
        print("Inflation Ratio(X,Y):", eval_part.eval_infl(args.norm));
        print("Refinement Error(X,Y):", eval_part.eval_re(args.norm));
    else:    
        print("Nestedness(X,Y):", np.sum(eval_part.eval_nest("global")));
        print("Inflation Ratio(X,Y):", np.sum(eval_part.eval_infl("global")));
        print("Refinement Error(X,Y):", np.sum(eval_part.eval_re("global")));
    #endif
    print(50*"-" + "\nRegion Cases\n" + 50*"-");
    if(args.norm == "global" or args.norm == "none" or args.norm == "regional"):
        print("Full Merge:", eval_part.count_fm(args.norm));
        print("Merge & Inflation:", eval_part.count_mi(args.norm));
        print("Full Inflation:", eval_part.count_fi(args.norm));
        print("Stability:", eval_part.count_stab(args.norm));
        print("Instability:", eval_part.count_inst(args.norm));
        print("Full Deflation:", eval_part.count_fd(args.norm));
        print("Deflation & Split:", eval_part.count_ds(args.norm));
        print("Full Deflation:", eval_part.count_fs(args.norm));
    else:    
        print("Full Merge:", np.sum(eval_part.count_fm("none")));
        print("Merge & Inflation:", np.sum(eval_part.count_mi("none")));
        print("Full Inflation:", np.sum(eval_part.count_fi("none")));
        print("Stability:", np.sum(eval_part.count_stab("none")));
        print("Instability:", np.sum(eval_part.count_inst("none")));
        print("Full Deflation:", np.sum(eval_part.count_fd("none")));
        print("Deflation & Split:", np.sum(eval_part.count_ds("none")));
        print("Full Deflation:", np.sum(eval_part.count_fs("none")));
    #endif
#end main

if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument("--x", help="Path to the label image of the first partition", type=str);
    parser.add_argument("--y", help="Path to the label image of the second partition", type=str);
    parser.add_argument("--norm", help="Normalization factor: none, regional, global", type=str);

    args = parser.parse_args();
    main(args);
#endif
