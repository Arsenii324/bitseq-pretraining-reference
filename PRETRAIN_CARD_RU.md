# Карточка предобучения BitSequence — фактическая спецификация

Дата упаковки:2026-10-07. Ниже определения, параметры, действия и наблюдения.
Интерпретация и границы вывода вынесены в [ASSESSMENT_RU.md](ASSESSMENT_RU.md).
Источники каждого численного результата — соседние `eval_*.json`, `CALIBRATION.json`,
`manifest.json`, `train.jsonl`, tensor-checkpoint и `source_snapshot.zip` в `artifacts/`.
Полные значения, матрица расстояний и SHA checkpoint’ов — [facts.json](facts.json).

## 1. Что именно обучалось и зачем

Обучался один тип **двунаправленного масочного предиктора**: по частично видимой
битовой последовательности предсказывать распределение каждого закрытого слова.
Три принятые модели различаются инициализацией/RNG и, для100, историей выбора
продолжения и внутренним возрастом Adam. Не обучались TraFL, TB, JustGRPO,
LoRA-адаптер или partition-head. Никакого RL на этом этапе нет.

Модель должна приближать известный многомодальный закон данных. Затем её реальные
веса замораживаются как reference для отдельных пост-тренировочных экспериментов.
Точный teacher для предобучения и фактический замороженный reference — разные объекты.

Условия задачи фиксированы; переменного текстового запроса нет. Единственный
начальный контекст — BOS. Обозначение `y|x` здесь означает эту единственную задачу,
а не распределение по множеству языковых запросов.

## 2. Пространство задачи и восемь мод

| Константа | Значение |
|---|---|
| Длина в битах |12 |
| Длина слова |2бита |
| Число позиций слов, L |6 |
| Словарь результата |0,1,2,3 → соответственно00,01,10,11 |
| Порядок битов | big-endian; слово0 содержит первые два бита |
| Полные допустимые результаты |4⁶=4096, без запрещённых комбинаций |
| Частичные состояния |5⁶=15625; каждая позиция — слово либо маска |
| Рёбра раскрытия |75000, каждое раскрывает одно слово |
| Exit/sink | технический завершающий переход framework, не дополнительное слово |

Моды не генерировались случайно при обучении; передавались явно в `H` исходного
`NonAutoregressiveBitSequence`. Их индексы ниже используются и в массивах basin masses.

| Индекс |12бит |6двухбитовых слов |
|---:|---|---|
|0|000000000000|0 0 0 0 0 0|
|1|101010110101|2 2 2 3 1 1|
|2|011001101100|1 2 1 2 3 0|
|3|110011011001|3 0 3 1 2 1|
|4|000111100011|0 1 3 2 0 3|
|5|101101010110|2 3 1 1 1 2|
|6|011110001111|1 3 2 0 3 3|
|7|110100111010|3 1 0 3 2 2|

Конструкция: для u∈{0,…,7} бит i равен parity(u AND cᵢ),
`c=[1,2,3,4,5,6,7,1,2,3,4,5]`. Расстояние — **число различающихся битов**,
не число разных слов, не строковое edit distance.

По28 неупорядоченным парам различных мод: минимум6, максимум8, среднее48/7≈6.857143.
Распределение расстояний:6 —8пар,7 —16пар,8 —4пары.

| Межмодовое расстояние |0|1|2|3|4|5|6|7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|0|7|6|7|6|7|8|7|
|1|7|0|7|6|7|6|7|8|
|2|6|7|0|7|8|7|6|7|
|3|7|6|7|0|7|8|7|6|
|4|6|7|8|7|0|7|6|7|
|5|7|6|7|8|7|0|7|6|
|6|8|7|6|7|6|7|0|7|
|7|7|8|7|6|7|6|7|0|

`d(y,M)=min_m d_Hamming(y,m)`. Число результатов на расстоянии d=0,1,2,3,4,5
от ближайшей моды соответственно8,96,528,1600,1672,192.

**Окрестность моды** (в других документах «basin») здесь — множество
`B_m={y:d_Hamming(y,m)≤2}`. Это не выявленный ландшафтный бассейн оптимизатора.
Каждая такая окрестность содержит79результатов:1+12+66. Они не пересекаются;
объединение содержит632результата. Метрика массы окрестности учитывает каждый
результат своим вероятностным весом, не просто наличие моды среди sampled outputs.

## 3. Законы данных и награды: показатели не смешиваются

| Объект | Формула и показатель | Где используется |
|---|---|---|
| Данные предобучения | q_pre(y)=exp(-12·d/12)/Z_pre=exp(-d)/Z_pre | IID терминальные примеры, teacher conditionals, калибровка |
| Награда среды/последующего RL | R(y)=exp(-24·d/12)=exp(-2d) | Оценка среднего reward; post-training, **не** loss предобучения |
| Нормированный reward-target | q_R(y)=exp(-2d)/Z_R | Цель отдельного canonical TB; **не** данные предобучения |

В `task_distribution(graph, alpha)` параметр alpha задаёт `q`, но возвращаемый
массив `reward` всегда равен exp(-2d), в том числе при alpha12. Alpha12 и
`reward_exponent=24` одновременно существуют без противоречия. Показатель
в hard→soft→tail **не менялся**. Последующий β TraFL — иной параметр,
не reward_exponent и не alpha учителя; здесь β вообще не используется.

| Точное свойство | q_pre∝exp(-d), alpha12 | q_R∝exp(-2d), exponent24 |
|---|---:|---:|
| log нормировки |5.422083354534|3.561002283548|
| Масса каждой B_m |.063382678830|.108892711495|
| Сумма масс восьми B_m |.507061430642|.871141691961|
| E[R], где R=exp(-2d) |.063159331414|.282553423460|

Teacher вычисляется перечислением всех4096результатов; full support, без
sampling-аппроксимации его вероятностей. H(q_pre)=7.835033894122нат.
Все восемь мод представлены в teacher/data. Отдельная test-выборка с новыми
модами или новые задачи не используются.

## 4. Сэмплирование данных, маски и loss

На каждом обновлении:

1.256 независимых Y из q_pre, с возвращением (`torch.multinomial`, CPU64 probabilities).
2. Для каждого Y выбрать l равномерно из{1,…,6}; затем равномерное подмножество
   A⊂{0,…,5} размера l. Одна новая непустая маска на пример.
3. Открытые слова оставить; позиции A заменить MASK; BOS всегда открыт.
4. Один forward по batch; вычислить loss ниже; backward, clipping, AdamW step.

Вероятность конкретной маски `P(A)=1/[6·C(6,|A|)]`. Возможных непустых масок63,
но **на обучении не перечислялись все63** и это не uniform-over63. Полная
маска встречается с вероятностью1/6. Среднее число закрытых слов3.5;
маргинальная вероятность закрыть фиксированную позицию7/12.
Не использовались complementary/antithetic masking, K=4, adaptive proposal или
повышенный K: это другие, последующие RL/диагностические опыты.

Обозначим `πθ(v|C,i)` выходной four-class softmax для позиции i при видимом
контексте C=(BOS,Y_не_A). Два оценщика **одной и той же ожидаемой функции**:

**Hard-label stage:**

```
L_hard = mean_batch [ -(6/l) Σ_{i∈A} log πθ(Y_i|C,i) ].
```

**Teacher-soft stage:**

```
T_i(v|C) = q_pre(Y_i=v | Y_не_A), точно вычислено по4096 Y.
L_soft = mean_batch [ (6/l) Σ_{i∈A} -Σ_{v=0}^3 T_i(v|C) log πθ(v|C,i) ].
```

Teacher-target detach; gradient только в πθ. Soft-loss аналитически интегрирует
закрытые **метки**, но видимые контексты и маски остаются случайными. Использование
маргинальных T_i корректно для суммы cross-entropy; оно не требует независимости
закрытых слов в совместном teacher-распределении. Масштаб6/l — extensive denoising
NELBO, не деление sequence loss на6 как в TraFL-surrogate.
Teacher-soft не означает soft/parallel decoding и не является новым sampler’ом.
Loss не является exact terminal negative log-likelihood выбранного декодера.

Data generator seed = model seed+10000; mask generator seed = model seed+20000.
Два CPU RNG и их состояния продолжаются между этапами. Это не on-policy sampling
модели, не reward-rejection sampling и не группы G=5.

## 5. Backbone, численная точность, оптимизация

Единственный backbone — специализация `tdlm.model.MaskPredictor`,
`bitseq.model.make_denoiser(seed)`. **398980 обучаемых скаляров,30 parameter tensors**.

| Компонент | Фактическая конфигурация |
|---|---|
| Вход | BOS5 +6слов; MASK4, словарь6, max_len7 |
| Выход | Linear128→4; не может выдавать BOS/MASK |
| Encoder |2 pre-LayerNorm bidirectional TransformerEncoderLayer |
| Attention |4heads, width128; без causal attention mask |
| FFN |512, GELU |
| Embeddings | обучаемые token и position |
| Финальная нормировка | LayerNorm |
| Dropout |0 |
| Инициализация | Linear/Embedding Normal(std=.02), Linear bias0; attention in_proj Xavier отдельно по двум слоям, bias0; штатная LayerNorm |
| Precision/device | float32 forward/backward, Apple M2 Pro MPS; без AMP/quantization |
| Trainability | все параметры; не LoRA |
| Optimizer | AdamW, β₁=.9, β₂=.999, eps=1e-8, weight_decay0 |
| Gradient clipping | общий L2 norm≤1, nonfinite error |
| LR | hard/soft-high6e-4; tail/soft-low6e-5 |
| Schedule | constant внутри этапа, один переход LR; без warmup/cosine |

Это не LLaDA-8B архитектура в уменьшенном масштабе: learned positions, LayerNorm
и стандартный PyTorch encoder — явные отличия. Другой backbone в этих опытах
предобучения не пробовали. Нет backward policy, reference network и logZ head
в pretraining loss. Teacher q_pre — закон данных, не backward policy.

## 6. Реальная история обучения:100 и фиксированные101/102

«Реальные обновления» ниже — суммарные изменения модели по её родительской цепочке;
не суммарная работа отброшенных соседних ветвей. «Adam clock» — прочитанное из
tensor state поле `step` для всех30 параметров, не число строк текущего журнала.

| Модель/этап | Новых updates | Loss | LR | Кумулятивные updates | Adam clock |
|---|---:|---|---:|---:|---:|
|100 исходный |6000|hard|6e-4|6000|6000|
|100 diagnostic hard-high |2000|hard|6e-4|8000|8000|
|100 diagnostic hard-low |2000|hard|6e-5|8000|10000|
|100 diagnostic soft-high, выбранный |2000|soft|6e-4|8000|12000|
|100 diagnostic soft-low |2000|soft|6e-5|8000|14000|
|100 extension от soft-high |4000|soft|6e-4|12000|16000|
|100 tail от extension, принятый |500|soft|6e-5|12500|16500|
|101 hard |6000|hard|6e-4|6000|6000|
|101 soft-high |6000|soft|6e-4|12000|12000|
|101 soft-low, принятый |500|soft|6e-5|12500|12500|
|102 hard |6000|hard|6e-4|6000|6000|
|102 soft-high |6000|soft|6e-4|12000|12000|
|102 soft-low, принятый |500|soft|6e-5|12500|12500|

У100 первая стадия наблюдалась каждые500, диагностики каждые250, eligibility
начиналась с500; extension каждые500. Tail имел максимум2000 и остановку на
первом прошедшем все gates наблюдении, фактически500. Все четыре diagnostic-ветви
начинались с одинаковых **весов** hard6000 и одинаковых data/mask RNG; Adam clock
был неодинаковым. Родительский checkpoint на диске не изменялся.

У101/102 заранее фиксировалось6000hard+6000soft-high+500soft-low, без спасательного
продолжения/выбора раннего endpoint. Наблюдения: initial/final hard, начальные
снимки soft и оценка в конце каждого soft-stage; принятие только по tail500.
Модели независимо инициализированы и обучены, не копии100.

Для каждой принятой модели фактическая родительская цепочка содержит12500×256=
**3200000 терминальных draws**. Это не число уникальных4096Y. Старое `samples`
в metadata100 равно1664000: оно считает только soft-chain6500×256, не весь ancestry.
Metadata101/102 содержит3200000. Исходные счётчики сохранены без переименования.

С учётом **отброшенных** трёх diagnostic-ветвей100 реально выполнено18500updates
и4736000draws в его кампании. Вместе с двумя независимыми12500-run’ами:
43500updates/11136000draws. Это стоимость описанной кампании предобучения,
не стоимость только трёх принятых родительских цепочек и не весь бюджет проекта
(предварительные smoke/timing/gradient checks отдельно). Phantom Adam+4000
не добавляется к этим числам как выполненная работа.

Для четырёх historical diagnostic-arm’ов фактические доли clipped gradients:
hard-high .9995, hard-low1.0, soft-high0, soft-low0; средние preclip norms
1.882996/1.926318/.217558/.208054 соответственно. Это наблюдения из
`artifacts/BS-calibration-diagnostic/results.json`, не чистое установление причины.

### Проверенный дефект Adam100

Исторический `calibration_continue.py` один раз загружает parent optimizer dictionary
и передаёт его последовательно четырём fresh optimizer’ам. Torch2.13 при
capturable=False/fused=None сохраняет ссылку на CPU-тензор `step`.
Предыдущая ветвь увеличивает этот общий clock in-place. CPU moments при переносе
на MPS копируются; дополнительные moment-updates отвергнутых ветвей **не** попадают
в следующий model lineage. Soft-high получает phantom+4000 от двух прежних hard-arm’ов;
extension/tail наследуют этот offset через сохранённые файлы.

Это подтверждено исходным архивом,19checkpoint’ами и отдельным минимальным MPS
воспроизведением; [аудит](evidence/PRETRAIN_OPTIMIZER_LINEAGE_REVIEW.md) даёт точные SHA
и границы воздействия. Нельзя назвать четыре ветви same-Adam контролем или100/101/102
одинаковым age-normalized рецептом. Нет corrected12500-run для seed100.
Фиксированный driver101/102 заново десериализует parent и проверяет фактические clocks.

## 7. Декодеры и точное измерение

**AR:** раскрыть позиции0,1,…,5, в каждой sampled word из model softmax.
Attention остаётся bidirectional; AR относится к порядку раскрытия, не backbone.

**Random order:** каждый шаг uniform по ещё закрытым позициям, затем sampled word
из её softmax. Не confidence order, не параллельное раскрытие, temperature1,
без top-k/top-p отсечения. Для каждого результата существуют720порядков.
Это конечные адаптации для проверки калибровки, не заявление о sampler’е TraFL-8B.

Оценка перечисляет все15625 partial contexts, получает float32 model logits,
делает **CPU float64 softmax** и распространяет массу по всем рёбрам DAG.
У каждого слоя и terminal law проверяется сумма1; terminal renormalization нет.
Оценки scalar/4096 terminal probabilities не являются MC-оценками по n generations.
В training/sampling softmax float32; эта численная разница явно сохранена.

Определения:

- TV(p,q)=½Σ_y|p(y)-q(y)|.
- Масса B_m=Σ_{y∈B_m}p(y); minimum basin — минимум восьми масс.
- Reward=Σ_y p(y)exp(-2d(y,M)).
- H(Y)=−Σ_y p(y)log p(y), нат.
- H(σ|Y)=Σ_y p(y)[−Σ_σ p(σ|y)log p(σ|y)], нат;
  AR структурно0, random≤log720≈6.579251.
- Denoising excess/word=[−E_qpre ELBOθ−H(q_pre)]/6.
  Независимый второй расчёт: сумма context-weighted conditional KL к teacher,
  делённая на6. Это не perplexity на новой задаче и не fill accuracy.

## 8. Калибровка и фактические результаты

Все критерии должны выполниться одновременно; пороги при неудачах не ослаблялись:
TV(AR,q_pre)≤.05; TV(random,q_pre)≤.05; относительная ошибка **каждой** массы B_m≤.20
при обоих декодерах; TV(AR,random)≤.02; denoising excess/word≤.02.

| Этап | TV AR→q_pre | TV random→q_pre | TV декодеров | Excess/word | Все gates |
|---|---:|---:|---:|---:|---|
|100 hard6000|.14417846|.11553905|.09096999|.01197004|нет|
|100 hard-high2000|.13214812|.10935935|.08327645|.01035727|нет|
|100 hard-low2000|.09940878|.07066356|.07328987|.00585239|нет|
|100 soft-high2000|.04374853|.02685293|.03384989|.00116649|нет|
|100 extension4000|.02745639|.01448265|.02372243|.00040556|нет|
|100 soft-low2000, не выбран|.07761842|.05376828|.05778795|.00388502|нет|
|100 tail500|.01834147|.00746203|.01709014|.00019225|да|
|101 hard6000|.14758947|.12375012|.08552064|.01199652|нет|
|101 soft-high6000|.02510588|.01538858|.02220103|.00038667|нет|
|101 tail500|.01763677|.00674441|.01648441|.00017653|да|
|102 hard6000|.14899669|.13184127|.09127368|.01351553|нет|
|102 soft-high6000|.02926299|.01820756|.02420508|.00050273|нет|
|102 tail500|.01948330|.00832509|.01842687|.00023656|да|

Soft-high100 до tail проходил не все gates: decoder agreement — реальный failure,
не устаревшее опасение. Данные таблицы — отдельные fitted instances, не mean±SE.
Полные восемь basin masses и их проверки доступны для каждого final в CALIBRATION.json.

| Принятые модели, random decoder |100|101|102|
|---|---:|---:|---:|
| E[R=exp(-2d)] |.06338625|.06320560|.06324966|
| minimum B_m mass |.06292301|.06296245|.06286906|
| Сумма восьми B_m |.50643967|.50602608|.50593226|
| H(Y), нат |7.83642091|7.83746726|7.83785896|
| E_current H(σ|Y), нат |6.57831113|6.57836782|6.57810965|

## 9. Identity, источники и границы пакета

Final SHA256:

```
100 788e00b95e0574716c15893a25205a1a84c0eb5ddf088d4ab392add41bc8c674
101 1a18eb5d190c820593072fdcf8649cb179ed66b54b654d39d31f82ac08f44d79
102 5548baadbe12b9e1f3724a2658e54cd82ff4d92678ff8619c388e1a7230f38a3
```

Каждый final-файл около4.84MB, содержит model state_dict, Adam moments/clock,
data/mask RNG, CPU/MPS RNG, metadata, не pickled model object.
Загружать с `weights_only=True`; архитектуру создаёт приложенная factory.

Framework torchgfn commit `f39bfc96407c0dcf443a41e5abb97503d707ea94`;
72Python-файла сверены с setup-manifest и vendored byte-for-byte.
Класс среды и его reward не переписаны. Независимый finite-graph/teacher/evaluator
— собственные диагностические реализации, не обучение из exact terminal likelihood.
Внешний JustGRPO source snapshot commit `1a2fddb5c6655597e63081c0af5ebb718a849f39`
включён для сохранения source identities; предобучение его loss не вызывает.

Это выбранная малая **any-order BitSequence адаптация**, не оригинальный120-bit
append-only benchmark TB с60модами, иной геометрией/порогами и50000updates.
Предобученные reference не предоставлены авторами TraFL/JustGRPO.
Основной downstream RL report в этот пакет не выдаётся за результат предобучения.
Подробная проверяемость/платформенная граница — [REPRODUCE_RU.md](REPRODUCE_RU.md).

Для проверки конкретной части спецификации:

| Вопрос | Код/первичный источник |
|---|---|
| Native task/reward | [неизменённый класс](vendor/gfn/gym/bitSequenceNonAutoregressive.py), [pinned upstream](https://github.com/GFNOrg/torchgfn/tree/f39bfc96407c0dcf443a41e5abb97503d707ea94) |
| Наши modes/config | [env.py](source/research/bitseq_mps/bitseq/env.py) |
| Exact q_pre/teacher | [oracle.py](source/research/bitseq_mps/bitseq/oracle.py) |
| Backbone/factory | [model.py](source/src/tdlm/model.py), [специализация](source/research/bitseq_mps/bitseq/model.py) |
| Маски/hard/soft loss | [train.py](source/research/bitseq_mps/bitseq/train.py), [mask/scoring helpers](source/src/tdlm/mdm.py) |
| Hard loop/gates | [run.py](source/research/bitseq_mps/bitseq/run.py) |
| Fixed101/102 stages/parent checks | [reproduce_base.py](source/research/bitseq_mps/checks/reproduce_base.py) |
| Exact laws/entropy/excess | [evaluate.py](source/research/bitseq_mps/bitseq/evaluate.py) |
| Original TB comparison boundary | [paper2201.13259](https://arxiv.org/abs/2201.13259) |

Это точные места для проверки, не список дополнительных обязательных глав.
