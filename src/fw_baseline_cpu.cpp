#include "fw_baseline_cpu.h"

void fwBaselineCPU(WeightType* D, int n, int tileSize) {
    (void)tileSize;

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const WeightType newPath = D[i * n + k] + D[k * n + j];
                if (newPath < D[i * n + j]) {
                    D[i * n + j] = newPath;
                }
            }
        }
    }
}
