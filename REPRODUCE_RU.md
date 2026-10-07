# Использование и воспроизводимость

## 1. Проверка файлов без Torch

Из корня нового репозитория:

```bash
python3 verify.py
python3 -m unittest discover -s tests -v
```

Inventory фиксирует файлы исходного release, не является cryptographic signature
от третьей стороны. SHA проверяет corruption/изменение относительно manifest, не
истинность научного вывода. `.git`, `__pycache__` и явно исключённые рабочие output
не входят в inventory. Исходные исторические JSON/ZIP/PT не перезаписываются.

## 2. Среда: что использовано и что не проверено

Фактическая среда обучения — Python3.11, Torch2.13.0, NumPy1.26.4 на macOS/M2Pro,
float32 MPS. Зависимости-оверлей записаны в
`source/research/bitseq_mps/setup_manifest.json`. Vendored gfn содержит именно
pinned72Python files, не предположение что любая PyPI-версия2.4.1 эквивалентна.
Остальные third-party dependencies не vendored. Нужны, в частности, einops,
tensordict, torch-geometric, psutil и их зависимости (у gfn есть eager imports).

`requirements-observed.txt` фиксирует наблюдавшиеся версии overlay и Torch/NumPy,
**не** гарантирует доступность wheels на любой платформе. Чистая установка всех
версий заново не проверена. Не заменять версии рабочего shared venv без необходимости.
Если среды нет — использовать отдельную venv, не глобальную pip-установку.

Местные validation commands использовали существующий read-only interpreter и
оригинальный overlay **только для зависимостей**; import bitseq/tdlm/gfn берётся
из этого export. Путь к прошлому ноутбуку не является переносимым требованием.
Training resource-monitor использует macOS sysctl и не работает на Linux без
отдельного адаптера. CPU загрузка/finite evaluation — другое утверждение, чем
Linux trainer portability. Не отключать monitor молча, чтобы заставить run работать.

## 3. Загрузка и переоценка трёх моделей

Для совместимой среды:

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/vendor:$PWD/source/research/bitseq_mps:$PWD/source/src${PYTHONPATH:+:$PYTHONPATH}"
python verify.py --neural --device cpu
BITSEQ_BASE_REPLAY_GATE=1 python -m pytest source/research/bitseq_mps/tests tests -q
```

Команда сначала проверяет inventory и архивные source pins, затем Adam clocks
всех13конечных этапов (по30state-tensors) и каждый принятый final
model state_dict, все15625контекстов,4096terminal laws, calibration
и расхождение с сохранённой MPS-таблицей. Модель не обновляется; RNG/optimizer
state не продолжаются. Предусмотрен явный допуск CPU↔MPS logword-table1e-4,
scalar1e-5, terminal-mass1e-6; actual differences выводятся, не скрываются за pass.
Fresh validation receipt в `evidence/VERIFICATION.json` содержит реально полученные
числа и границы; не трактовать допуск как измеренную погрешность.

Пять исходных validation-файлов в `source/research/bitseq_mps/tests` добавлены
при упаковке (SHA/роль в `evidence/VALIDATION_TEST_PROVENANCE.json`), они не часть
исторического source ZIP. Проверяют exact teacher/DAG против brute paths,
hard/soft loss/gradient, CPU/MPS primitives, checkpoint isolation/continuation,
реальный short hard→soft→tail replay. Это не новое12500-step обучение.
На машине без MPS device-test будет skipped, не засчитан как passed. Один driver
smoke использует macOS resource monitor, поэтому весь этот suite не обещан дляLinux.

Минимальная загрузка одной модели:

```python
import torch
from bitseq.model import make_denoiser

checkpoint = torch.load(
    'artifacts/BS-independent-base101/soft-low/ckpt_000500.pt',
    map_location='cpu', weights_only=True,
)
model = make_denoiser(101)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
logits = model(torch.tensor([[5,4,4,4,4,4,4]]))[:,1:]
assert logits.shape == (1,6,4)
```

Для RL-reference фиксировать эти фактические веса и полный SHA, не подставлять
q_pre вместо p0. Для новой optimizer-ветви десериализовать state заново или
deepcopy, а не переиспользовать dictionary между ветвями. Счётчик100 не «чинить»
при read-only evaluation; если намеренно продолжать его, отметить Adam16500.

## 4. Повторение fixed recipe101/102 — не запуск автоматически

Байты исходных членов actual101 source ZIP в `source/` сохранены неизменёнными;
дополнительные validation tests обозначены отдельно. Полный layout
сохранён потому что original driver вычисляет ROOT из относительного положения.
Проверка плана не обучает модель и не создаёт checkpoint:

```bash
python source/research/bitseq_mps/checks/reproduce_base.py plan \
  --seed 101 --out "$PWD/work/base101" --storage-root "$PWD/work"
```

При явном желании выполнить12500updates на совместимом **Mac**:

```bash
python -u source/research/bitseq_mps/checks/reproduce_base.py run \
  --seed 101 --out "$PWD/work/base101" --storage-root "$PWD/work" --device mps
# Аналогично seed102; другой несуществующий output, не перезапись первого.
```

Driver: hard6000/soft-high6000/soft-low500, batch256, final-only eligibility;
при fail нет выбора «самого близкого» checkpoint. Source pin refusal, фактические
parent Adam clocks, SHA и cumulative counters проверяются. Storage cap2GiB для
указанного fresh storage-root; до run нужны≥10GiB свободно, normal pressure,
без роста swap>1GiB, RSS≤2GiB. Один собственный neural job одновременно.
Битовая воспроизводимость исходного long MPS run не обещана. Новая12500-step
реплика в ходе **упаковки** не запускалась; уже существующие101/102 — реальные runs.

Исторический100 diagnostic entrypoint находится в source ZIP кампании, а не
предлагается как safe new training entrypoint. Его последовательность/alias bug
нужны для аудита старой модели, не для повторения якобы строгого four-arm experiment.
Исторические комментарии/протоколы внутри source ZIP могут предшествовать обнаружению
этого дефекта: актуальная карточка и аудит имеют приоритет над их same-recipe формулировками.

## 5. Происхождение и лицензии

Original source ZIPs не удалены после распаковки; parent references в старых
manifest могут быть абсолютными путями старого repo, тогда как данные здесь
лежат под `artifacts/`. SHA, а не имя пути, связывает родительские bytes.
В инвентаре нет W&B/HF/Kaggle/GitHub credentials или venv.

`licenses/torchgfn-LICENSE` — фактическое короткое upstream Apache-2.0 уведомление;
полный [текст Apache-2.0](licenses/Apache-2.0.txt) добавлен отдельно из
[официального источника](https://www.apache.org/licenses/LICENSE-2.0.txt).
Original notice не заменён. Его package
metadata говорит MIT: discrepancy оставлено открытым, не разрешено произвольным
перелицензированием. JustGRPO notice/LICENSE сохранены отдельно. Наш собственный
код/документы здесь не заявлены как новый blanket license для всего дерева.
