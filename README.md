# neon_gui — GUI для neon_test

Кроссплатформенный GUI поверх `neon_test.cpp` на **ImGui + ImPlot**.

---

## Структура файлов

```
neon_gui.cpp        ← GUI-код (этот файл)
CMakeLists.txt      ← сборка на всех платформах
imgui/              ← исходники ImGui (https://github.com/ocornut/imgui)
implot/             ← исходники ImPlot (https://github.com/epezent/implot)
```

---

## Сборка

### Linux / ARM (Raspberry Pi, Jetson, …)

```bash
# Зависимости (Ubuntu / Debian)
sudo apt install libglfw3-dev libgl1-mesa-dev cmake g++

# Клонировать ImGui + ImPlot рядом с neon_gui.cpp
git clone https://github.com/ocornut/imgui
git clone https://github.com/epezent/implot

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/neon_gui
```

### Windows (MSVC + vcpkg)

```bat
vcpkg install glfw3 opengl
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
build\Release\neon_gui.exe
```

### Windows (MinGW)

```bat
pacman -S mingw-w64-x86_64-glfw mingw-w64-x86_64-cmake
cmake -B build -G "MinGW Makefiles"
cmake --build build
```

### Android (NDK + SDL2)

```bash
# Добавьте в CMakeLists.txt путь к SDL2 и соберите через Android Studio
# или вручную:
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21 \
  -DSDL2_DIR=<path_to_SDL2>
cmake --build build
```

---

## Что показывает GUI

| Панель | Содержимое |
|---|---|
| **Platform Info** | Флаги NEON, режим AArch64 / ARMv7 |
| **Correctness Tests** | Таблица: 13 тест-кейсов, scalar / NEON / unrolled vs expected |
| **Custom Array** | Генератор массива с заданными N, seed, диапазоном, долей нулей |
| **Benchmark Control** | Запуск замеров в фоновом потоке, статус |
| **Benchmark Results** | Таблица: время (мс) и ускорение (x) для N = 1K / 10K / 100K / 1M |
| **Execution Time (ms)** | Линейный граф log–log: scalar / NEON / unrolled |
| **Speedup vs Scalar** | Линейный граф ускорения, линия паритета x=1 |
| **Time at N=1M (bar)** | Bar-chart трёх реализаций для максимального N |

---

## Зависимости

- **ImGui** ≥ 1.90 — https://github.com/ocornut/imgui  
- **ImPlot** ≥ 0.17 — https://github.com/epezent/implot  
- **GLFW** ≥ 3.3 (Linux / Windows) **или** **SDL2** ≥ 2.0 (Android)  
- OpenGL 3.3 Core (Linux/Win) / OpenGL ES 2.0 (Android)  
- C++17, pthreads