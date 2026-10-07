# Оценка предобучения и ограничения

Это выводы автора пакета, отдельно от [фактической карточки](PRETRAIN_CARD_RU.md).

## Что позволяет утверждать результат

Получены три реальные fitted reference-модели с известными весами и небольшим
расхождением от q_pre у двух заданных декодеров. Нужные для этой конечной задачи
вероятности измеряются точно по всему пространству. Поэтому последующая RL-работа
может сравнивать изменения относительно **фактического** p0, а не приписывать
предобученной сети идеальное teacher-распределение. Принятие не означает p0=q_pre.

Hard6000 недостаточен: все три экземпляра не прошли gate decoder agreement.
Soft-stage и tail дали проходящие checkpoint’ы, но это не доказательство, что
единственная причина — снижение label-gradient variance. Меняются число updates,
LR, clipping/Adam dynamics; диагностический четырёхветочный опыт100 дополнительно
имеет ошибку clock. Frozen-gradient расчёт hard/soft expectation/variance, не
использующий Adam, остаётся отдельным свидетельством, не causal convergence proof.

## Наиболее существенные ограничения

1. **Происхождение100 не эквивалентно101/102.**12500updates верны, Adam16500 тоже
   верен; нельзя свести различие между references только к seed. В source100 баг
   не исправлен в архивах. Новое matched исследование должно использовать fresh
   parent loads/deepcopy и gate clock/moment isolation. Нужен corrected seed100
   rerun, если требуется оценить именно эффект этого дефекта. Не выполнен.
2. **Задача существенно удобнее языковой генерации.** Одна симметричная кодовая
   конструкция, известные восемь мод, короткая длина, точный full-support teacher.
   Модель уже видела все моды. Выводы относятся к сохранению/перераспределению
   массы, не discovery неизвестных решений или reasoning generalization.
3. **Декодер определяет p(y).** AR и uniform-order — разные terminal laws одной
   сети; residual decoder TV<.02 не делает их идентичными. No confidence/parallel
   schedules здесь не проверялись. Bidirectional attention не заменяет проверку
   sampler’а из real-model pipeline.
4. **Нет претензии на scaled LLaDA или full TB benchmark reproduction.** Архитектура,
   длина, mode geometry/reward и trainer отличаются от оригинальных работ.
   Native torchgfn code conformance — другой, более узкий факт.
5. **Калибровка — операционное условие допуска.** Это не theorem о всех контекстах,
   величинах градиента или о downstream learned-Z/TraFL correctness. Пороги .05/.02
   заданы для данного исследования; другой вопрос может потребовать строже.
6. **Точная оценка ≠ точное обучение.** Training contexts/masks sampled; soft labels
   интегрированы точно. Eval CPU64(device32 logits) не bit-identical к native32
   sampling. Нет независимого full12500 retrain на Linux/CUDA или обещания совпадения
   длинного MPS training до последнего бита.

## Как трактовать Adam-дефект без преувеличения и сглаживания

Не «ещё4000updates» и не contamination чужими moment updates: это общий mutable
CPU clock в повторно использованном dict. При фиксированных одинаковых moments
и градиенте верхняя оценка изменения age-only bias-correction масштаба в начале
выбранной ветви около.1214%, затем меньше. **Это не bound на финальную модель**:
позднейшие градиенты меняются, нелинейный training может усилить возмущение.
Нет права объявить проблему научно несущественной по этой одной оценке.

При этом реальные сохранённые веса100 и их калибровочные таблицы остаются
измеренными объектами. Последующие RL optimizers создавались заново, не из
pretraining moments; frozen audits читали веса. Баг меняет историю и causal
утверждения, а не отменяет автоматически каждый downstream numerical result.

## Следующие исследования, а не содержимое этого пакета

- Исправленный seed100 по тому же fixed schedule и сравнение со старым fitted p0:
  изолировать clock-history только при контроле RNG/implementation variation.
- Другой codebook/асимметричные массы и новые моды: проверить зависимость от symmetry
  и заранее заданной multimodal поддержки.
- Longer L либо другой backbone: сначала новый exact/sanity budget, не молчаливое
  переносимое утверждение из L6.
- В основном локальном исследовании: matched terminal-tilt/path reference objectives,
  разные weighting populations conditional entropy/KL, stochastic-score controls.

Ни одно из этих предложений не выдано за уже выполненный результат предобучения.
