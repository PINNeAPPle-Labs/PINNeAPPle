- Medida confiável de memória de CPU (`pinneapple_analysis.cost.memory`, issue #54): o pico por alocador em vez do
  RSS amostrado como padrão, com o limite de cada método medido e documentado no módulo — `tracemalloc` é exato para
  Python e NumPy (as alocações do `PyDataMem` chegam ao tracemalloc) e cego para tensores; o profiler do PyTorch
  (`profile_memory=True`, somando `self_cpu_memory_usage` em ordem cronológica, nunca o inclusivo, que dobra o valor)
  é exato para tensores de CPU e cego para NumPy; os dois somados dão o pico vivo (`mem_source="torch+trace"`), com
  as partes em `mem_components`. O watermark do kernel (`getrusage().ru_maxrss`) só é reutilizável em processo novo,
  então `process` mede num filho com fork e devolve `None` — nunca um número — quando o filho não pode rodar autograd,
  que é o caso de todo passo de treino. `rusage_unit_bytes`/`to_bytes` decidem a unidade num só lugar: `ru_maxrss` é
  KiB no Linux e bytes no macOS, e o erro de ler um com o fator do outro é de 1024x (teste
  `test_ru_maxrss_unit_is_kib_on_linux_and_bytes_on_darwin`). O passo de PINN medido contra uma referência de
  workspace injetado fecha dentro dos 10% que a issue pede, e o número do método `process` é comparado ao que o
  kernel reporta a um processo separado (`test_process_peak_agrees_with_the_kernel_number_of_a_separate_process`).
  Um pico que não pôde ser medido é `bytes=None`, `source="unknown"`, e um orçamento não o aprova nem o reprova.
