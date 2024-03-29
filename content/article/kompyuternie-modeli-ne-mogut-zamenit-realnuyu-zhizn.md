---
title: "Компьютерные модели не могут заменить реальную жизнь"
author: "Питер Эрл"
date: 2020-08-26T13:00:00+03:00
tags: ["Питер Эрл"]
draft: false
---
![](https://www.aier.org/wp-content/uploads/2020/07/equationsAI-1536x975.jpg)

Прошло почти семь месяцев после принятия Декларации ВОЗ о чрезвычайной ситуации в области глобального здравоохранения. И хотя существует много дат, которые могли бы считаться началом кризиса, эта дата представляется особенно уместной: вскоре после нее эпидемиологи начали использовать количественные инструменты в своих аргументах, и именно после этого выводы моделей начали использоваться в политических схемах.

Теперь мы знаем, насколько ошибочными оказалось большинство из этих планов, не только с точки зрения прогнозирования, но и с точки зрения реальных последствий: потерянных жизней, богатства и возможностей. Когда эпидемиологические модели пытаются определить или предсказать уровень заражения некоего патогена, они еще могут быть полезны. Но в той степени, в которой эти прогнозы, в свою очередь, зависят от решений, принимаемых людьми --- часто путем включения характеристик агента, который копирует принятые решения --- эти усилия обречены на провал и являются упражнениями в самонадеянности.

Обзор таксономии проблем говорит об их невероятной сложности, как и сложности методов, используемых для их решения.

Я, прежде всего, экономист, а не математик, программист или физик-теоретик. И в связи с этим я решил не вводить  понятия полиномиального времени, недетерминированых машин и тому подобного, что обычно является частью такого обсуждения. Тем не менее, я полагаю, что сложность природы человеческих мыслей и действий перед лицом неопределенности или препятствий --- и тщетность попыток приблизить их в моделях или симуляциях --- станут очевидными.

## NP-полнота (NP-completeness)

Существуют некоторые проблемы, которые на первый взгляд кажутся простыми: например, положить большое количество объектов неправильной формы в большую коробку. Очевидно, что никакие два объекта не могут занимать одно и то же пространство одновременно. И хотя возможно приблизиться к оптимальному порядку нерегулярных объектов внутри коробки, но может быть чрезвычайно трудно определить оптимальное структурирование внутри коробки. И, кроме того, разница между упорядочением, близким к оптимальному и фактическим оптимальным упорядочением может быть небольшой, но нетривиальной: например, может потребоваться вытащить каждый из объектов неправильной формы и повторить процесс.

Может оказаться полезным думать о решениях проблемы такого рода как о трехмерной поверхности. Если есть 500 объектов неправильной формы, которые мы должны поместить в коробку --- скажем, что требование состоит в том, чтобы упорядочить все объекты таким образом, чтобы коробку можно было закрыть, --- и в коробке есть 50 нефиксированых мест с возможностью 10 различных ориентаций для каждого объекта -задача потребовала бы (500 × 50 × 10) 250000 битов, чтобы точно представлять все возможные упорядочения в коробке.

Теперь представьте, что каждое из этих возможных упорядочений оценивается по количеству свободного места, которое осталось в коробке, при этом отрицательное число соответствует ситуации, когда коробку закрыть невозможно, а положительное число говорит о том, что объекты организованы успешно. То есть, мы получаем своего рода трехмерный ландшафт: "провалы" или "долины" в одних местах, "холмы" и "горы" в других. Каждый "холм" представляет собой успешную организацию объектов внутри коробки, (коробка закрывается правильно), а оптимальным решением будет самая высокая "гора" на поверхности нашего решения.

Существуют различные алгоритмы, которые можно использовать для решения таких проблем, и одни из них более эффективны, чем другие. (Термин ["эффективность"](https://mc-stan.org/docs/2_22/stan-users-guide/statistical-vs-computational-efficiency.html) в вычислительной науке используется для описания времени, памяти или количества шагов, предпринятых для выполнения вычислительной задачи.) В рамках метафоры "поверхности решения" такие алгоритмы могут начинаясь в некоторой точке, итеративно отбрасывать определенные области поверхности, и быстро приводить нас к области ландшафта, которая содержит решение; и в конечном итоге, --- оптимальное решение.

Это возможно потому, что проблема такого типа имеет внутреннюю структуру. Существует порядок, который может быть простым или сложным, но тем не менее он существует; и учитывая, что он существует, с ним можно работать и решить с достаточной вычислительной мощностью. Мы можем не только найти лучшее решение проблемы, но и проверить решение и увидеть, что у нас получилось, повторив задачу снова.

## NP-сложность (NP-hardness)

Задача NP-hard либо не имеет такой структуры, либо структура настолько сложна, что может быть принципиально непроницаемой. В отличие от проблем в классе NP-Complete, такую ​​проблему не только трудно решить, но и чрезвычайно трудно определить, является ли выбранное решение лучшим.

Мало того. Доказательство того, что проблема на самом деле является NP-Hard, само по себе является проблемой NP-Hard.

## Вычислительная сложность в повседневной жизни

Вычислительная сложность является темой для магистрских курсов и курсов на уровне диссертации по информатике (а иногда и по физике). Проблемы, которые они изучают, выходят далеко за рамки помещения предметов неправильной формы в коробку: они в большей степени определяют существование оптимального результата для игры в [обобщенные шахматы](https://en.wikipedia.org/wiki/Generalized_game). Что касается сложности поиска решения, они имеют дело с решениями, которые занимают не часы или дни, а порой кратны теоретическому времени жизни вселенной для вычисления. Существует дискуссия о том, будут ли даже квантовые компьютеры полезны для решения самых сложных проблем NP-Hard.

Но этот класс сложных (часто почти невозможных) проблем, решение которых еще труднее (и, возможно, фактически невозможно найти решение), не относится к научной эзотерике: люди сталкиваются с ними ежедневно. Начало новой работы, принятие решений о сбережении или потреблении и множество других задач такого типа технически  являются задачами NP-hard, даже когда решения в них бинарны. Последовательность соображений, которые делают образ действий идеальным, скрыт неопределенностью и меняющейся ситуацией и попытка оглянуться назад, чтобы определить, был ли выбран правильный вариант, является бесполезным упражнением. Хотя мы часто думаем, что в прошлой ситуации, мы выбрали бесспорно лучшую альтернативу из всех, возможно, это не случилось иначе, чем по чистой случайности.

Многие проблемы NP-Hard, как в теории, так и в повседневной жизни, предполагают оптимизацию.

И я говорю "возможно, это не случилось иначе", потому что, как упоминалось ранее: доказательство того, что проблема относится к категории NP-Hard, само по себе NP-Hard. Я не могу доказать это, и даже если бы я мог, для меня и для кого-либо еще, было бы невозможно это подтвердить.

## Если так много неразрешимого, то как мы решаем хоть что-нибудь?

Возможно этот фрейминг звучит нигилистично, но тут нет никакого нигилизма; как нет его и в самой эпистемологии. Мало кто был бы удивлен, узнав, что есть некоторые вопросы, на которые никогда не будет четких ответов, или что разбор прошлых решений иногда сводит с ума. (И опять же, даже когда мы говорим себе, что сделали идеальный выбор в идеальное время и в идеальном месте, мы, скорее всего, участвуем в безобидном самообмане.)

Так как же люди решают проблемы? Во-первых, некоторые NP-Hard проблемы в основном решаемы с помощью простых, но мощных инструментов. Хотя многие из решений и проблем, с которыми мы сталкиваемся, технически могут быть NP-Hard ("Какую дипломную программу я должен выбрать?"), мы способны их решать. Хотя определение интеллекта остается неокончательным, миллионы лет эволюции привели к тому, что человеческий разум "загружен" эвристикой: мы накапливаем эмпирические правила, собираем прошлый опыт для построения оправданых догадок, и мы рождены со способностью провести базовый эмпирический анализ. Все это, кроме того, добавляется и оттачивается со временем и опытом.

Короче говоря, человеческий разум автоматически отбрасывает бесчисленные решения, которые либо явно неверны, либо не соответствуют другим требованиям нашего процесса поиска решений, и сосредотачивается только на том наборе решений, который у него остается. И все это время временные предпочтения, социальные и культурные факторы влияют на то, что мы, в конце-концов выберем. Более того, успех или неудача применения выбранных эвристик влияет как на то, какие из них мы выбираем, так и на порядок, в котором они используются для решения проблем в будущем, что также влияет на нашу способность решать проблемы.

## В чем здесь смысл?

Мы уже несколько месяцев наблюдаем, как применяются мощные количественные методы для интерпретации и прогнозирования быстро меняющихся обстоятельств пандемии. В этой среде отчетливо видны многочисленные проблемы социальных наук. Несмотря на то, что проблемы упрощения действий сотен миллионов или миллиардов людей до агентов в агентской модели или узлов и дуг в сетевой модели иногда признаются, этот метод информационной политики продолжает использоваться.

В уме каждого человека разрабатываются десятки проблем NP-Hard: некоторые практически мгновенно, а другие пошагово --- днями, годами или десятилетиями. В целом, расшифровка способов, которыми будут действовать огромные массы людей, аналогично является проблемой NP-Hard. Результат взаимодействия отдельных ответов на проблемы NP-Hard (и, разумеется, все другие проблемы в других категориях, с которыми сталкиваются люди) совершенно непредсказуем; столь же трудноразрешимый в совокупности, как в каждом из сотен миллионов или миллиардов умов.

Жизнь --- задача NP-Hard, и попытка выразить это мышление в нескольких строках кода неосмотрительна, но сама по себе, вероятно, безвредна. Однако, доведение этих приблизительных результатов до политиков, у которых нет ни склонности к скептицизму, ни побуждения действовать осторожно, опасно.

[Оригинал статьи](https://www.aier.org/article/computer-models-cant-substitute-for-real-life/)

> Перевод: Наталия Афончина

> Редактор: Владимир Золоторев
