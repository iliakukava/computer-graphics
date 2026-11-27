# Лабораторная №4, вариант 12

Карта теней от направленного источника света (солнца) на оценку 4. За основу взята третья лабораторная.

Что добавлено:

- генерация карты теней (`VK_FORMAT_D32_SFLOAT`, 2048x2048) каждый кадр через Dynamic Rendering (`vkCmdBeginRendering` / `vkCmdEndRendering`);
- отдельный графический конвейер для карты теней только с вершинным шейдером `shadow.vert`;
- сэмплер со сравнением глубины (`compareEnable = VK_TRUE`, `VK_COMPARE_OP_LESS`) и `sampler2DShadow` во фрагментном шейдере;
- в ImGui настраиваются площадь ортопроекции солнца и сдвиг глубины (`depth bias`).

Сборка из каталога `lab4`:

```bash
cmake --preset msvc-debug
cmake --build build-debug --parallel
```

Если собираете через MinGW/Ninja:

```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -S . -B build-debug
cmake --build build-debug --parallel
```

Запускать из корня `lab4`.
