# Floyd-Warshall GPU (CUDA)

Implementacija Floyd-Warshallovog algoritma na grafičkom procesoru (GPU) koristeći CUDA. Projekat uključuje tri verzije algoritma: osnovnu GPU verziju, verziju s klasičnim popločavanjem (tiling), te verziju s višeslojnim popločavanjem koja donosi značajna ubrzanja za velike grafove.

## Sistemski zahtjevi

- CUDA Toolkit 11.0+ (preporučeno 12.0+)
- C++ 17 kompajler (GCC 9+, Clang 10+, ili MSVC 2019+)
- CMake 3.18+
- NVIDIA GPU s Compute Capability 7.0+

## Build instrukcije

### Linux / Google Colab
```bash
git clone https://github.com/ddevedzic1/FW_CUDA.git
cd FW_CUDA
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
make -j$(nproc)
```

### Windows (Visual Studio)
```bash
cd FW_CUDA
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=75 ..
cmake --build . --config Release
```
> **Napomena:** Potrebno je prilagoditi `DCMAKE_CUDA_ARCHITECTURES` prema korištenom GPU-u.

Nakon uspješnog builda, izvršne datoteke se nalaze u `bin/` direktoriju.

## Dostupni algoritmi

### CPU verzija
| Naziv | Opis |
|-------|------|
| `baseline_cpu` | Osnovna sekvencijalna verzija za verifikaciju |

### GPU verzije
| Naziv | Opis |
|-------|------|
| `baseline_gpu` | Osnovna GPU verzija bez optimizacija |
| `tiling_gpu` | Klasično popločavanje |
| `multi_layer_tiling_gpu` | Višeslojno popločavanje |

## Pokretanje

### Benchmark Runner

Generiše graf, izvršava algoritam i mjeri vrijeme izvršavanja.

```
Sintaksa: bench_runner <algorithm_name> <graph_size> <density> [seed] [tile_size] [kappa]

Parametri:
  algorithm_name  - Naziv algoritma (vidjeti listu iznad)
  graph_size      - Broj čvorova grafa (max 32768)
  density         - Gustina grafa (0.0 - 1.0)
  seed            - (Opcionalno) Seed za generator, 0 = default
  tile_size       - (Opcionalno) Veličina pločice, default: 32
  kappa           - (Opcionalno) Parametar višeslojnosti za multi_layer_tiling_gpu
```

**Primjeri:**
```bash
# Osnovna GPU verzija
./bin/bench_runner baseline_gpu 4096 0.9

# Klasično popločavanje
./bin/bench_runner tiling_gpu 8192 0.9 42 32

# Višeslojno popločavanje s kappa=6
./bin/bench_runner multi_layer_tiling_gpu 8192 0.9 42 32 6
```

## Parametri algoritama

### Veličina pločice (tile_size)
- Fiksirana na **32** za optimalne performanse
- Podudara se s veličinom warpa na NVIDIA GPU-ima
- Omogućava potpuno odmotavanje petlji

### Parametar višeslojnosti (kappa)
- Koristi se samo za `multi_layer_tiling_gpu`
- Preporučene vrijednosti: **4-8**
- Veće vrijednosti smanjuju pristup globalnoj memoriji, ali povećavaju overhead
- Za κ=1 dobija se ekvivalent klasičnog popločavanja

### Tests Runner

Pokreće skup testova za verifikaciju ispravnosti algoritma.

```
Sintaksa: tests_runner <algorithm_name> [tile_size] [kappa]

Parametri:
  algorithm_name  - Naziv algoritma (vidjeti listu iznad)
  tile_size       - (Opcionalno) Veličina pločice
  kappa           - (Opcionalno) Parametar višeslojnosti za multi_layer_tiling_gpu
```

**Primjeri:**
```bash
# Osnovna CPU verzija
./bin/tests_runner baseline_cpu

# Osnovna GPU verzija
./bin/tests_runner baseline_gpu

# Višeslojno popločavanje s tile_size=32 i kappa=4
./bin/tests_runner multi_layer_tiling_gpu 32 4
```

## Struktura projekta

```
FW_CUDA/
├── include/           # Header datoteke
├── src/               # Izvorni kod
│   ├── fw_baseline_gpu.cu
│   ├── fw_tiling_gpu.cu
│   └── fw_multi_layer_tiling_gpu.cu
├── bin/               # Izvršne datoteke (nakon builda)
├── profiler/          # Rezultati profilisanja (CSV)
└── CMakeLists.txt
```