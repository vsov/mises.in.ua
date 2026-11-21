# Figure 2 recovered and verified

Article: content/article/indias-economic-miracle-lifted-500-million-out-of-poverty.md

The original HTML still uses the same main src for Figure 1 and Figure 2:
https://thedailyeconomy.org/wp-content/uploads/2026/11/image-11.png

Recovered on 2026-11-21 from Figure 2's distinct srcset:
https://thedailyeconomy.org/wp-content/uploads/2026/11/image-11-768x460-1.png

Publisher article:
https://thedailyeconomy.org/article/indias-economic-miracle-lifted-500-million-out-of-poverty/

Figure 1 uses WordPress attachment 43404; Figure 2 uses attachment 43405.
The latter's media API lists the recovered image as its medium_large size:
https://thedailyeconomy.org/wp-json/wp/v2/media/43405

Verified all seven bar labels independently against India (IND), gdppc,
Full data sheet, Maddison Project Database 2023:
https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2023
https://dataverse.nl/api/access/datafile/421302

Calculation: 100 * (gdppc[end_year] / gdppc[start_year] - 1).

| Period | Calculated percent | Chart label |
| --- | ---: | ---: |
| 1950-1959 | 15.8055 | 15.8 |
| 1960-1969 | 12.2500 | 12.3 |
| 1970-1979 | 3.1069 | 3.1 |
| 1980-1989 | 35.3846 | 35.4 |
| 1990-1999 | 29.7432 | 29.7 |
| 2000-2009 | 55.1244 | 55.1 |
| 2010-2019 | 61.1666 | 61.2 |

The publisher's caption "1969-1979" is incorrect: that interval gives
5.9391%, not 3.1%. The chart uses calendar-decade endpoints, nine elapsed
years per bar (ten annual observations). The article caption now describes
the actual coverage and an editorial note clarifies the interval error.
No chart was redrawn and no data were substituted.

## Subsequent correction: Figure 1 was also wrong

The initial localized full-size image-11.png was itself the Figure 2
bar chart, not the Figure 1 scatterplot. Replacing only Figure 2 left
two visually equivalent bar charts. Earlier verification of distinct
filenames and hashes did not establish distinct chart contents.

Recovered Figure 1 from attachment 43404's separate srcset derivative:
https://thedailyeconomy.org/wp-content/uploads/2026/11/image-11-768x460.png

Visual inspection confirms a scatterplot: GDP-per-capita growth on the
horizontal axis, poverty-rate change on the vertical axis, a descending
dotted trend line. This matches the Figure 1 caption and surrounding text.
Figure 2 remains the independently verified India bar chart. Both captions
are now in Russian; World Bank and Maddison remain distinct sources because
the figures use distinct data. No independent country-by-country
reconstruction of Figure 1 has been performed.

The untracked content/article/image-7.png also visually matches the recovered
chart at a larger size, but its origin is not established; it is not used
or modified. The publisher-hosted derivative provides traceable provenance.
