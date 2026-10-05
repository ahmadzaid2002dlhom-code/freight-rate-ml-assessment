# Unseen categories and inference robustness

Reproduce with `.venv/bin/python scripts/check_unseen_categories.py`. Coverage uses all 48,000 labeled development rows and all 12,000 unlabeled final-validation rows. Final-validation data is used only to describe coverage and test prediction compatibility; no accuracy metrics, tuning, or model selection use it.

## City coverage

| Coverage | Development | Final validation |
| --- | --- | --- |
| Distinct pickup labels | 64 | 72 |
| Distinct delivery labels | 64 | 72 |
| Distinct cities across both roles | 64 | 72 |
| Distinct directed routes | 4014 | 4214 |
| Distinct equipment labels | 3 | 3 |

**Equipment labels:** development and final validation both contain Dry Van, Flatbed, and Reefer; no new equipment label occurs in the supplied final inputs. Synthetic inference tests cover a new equipment label as well.

**Cities seen in development:** Albany, Albuquerque, Amarillo, Atlanta, Austin, Bakersfield, Baltimore, Baton Rouge, Birmingham, Boston, Buffalo, Charleston, Chattanooga, Cincinnati, Columbia, Corpus Christi, Dallas, Dayton, Detroit, El Paso, Fort Wayne, Fresno, Grand Rapids, Green Bay, Greensboro, Harrisburg, Hartford, Houston, Indianapolis, Jacksonville, Kansas City, Las Vegas, Lexington, Little Rock, Los Angeles, Louisville, Lubbock, Madison, Memphis, Milwaukee, Mobile, Montgomery, Nashville, New Orleans, New York, Oklahoma City, Philadelphia, Phoenix, Providence, Raleigh, Reno, Richmond, Salt Lake City, San Antonio, San Francisco, Savannah, Shreveport, St. Louis, Syracuse, Tampa, Toledo, Tucson, Tulsa, Washington.

**Cities seen in final validation:** Albany, Albuquerque, Allentown, Amarillo, Atlanta, Austin, Bakersfield, Baltimore, Baton Rouge, Birmingham, Boston, Buffalo, Charleston, Charlotte, Chattanooga, Chicago, Cincinnati, Columbia, Corpus Christi, Dallas, Dayton, Detroit, El Paso, Fort Wayne, Fresno, Grand Rapids, Green Bay, Greensboro, Harrisburg, Hartford, Houston, Indianapolis, Jackson, Jacksonville, Kansas City, Knoxville, Laredo, Las Vegas, Lexington, Little Rock, Los Angeles, Louisville, Lubbock, Madison, Memphis, Milwaukee, Mobile, Montgomery, Nashville, New Orleans, New York, Norfolk, Oklahoma City, Philadelphia, Phoenix, Providence, Raleigh, Reno, Richmond, Salt Lake City, San Antonio, San Diego, San Francisco, Savannah, Shreveport, St. Louis, Syracuse, Tampa, Toledo, Tucson, Tulsa, Washington.

**Completely new cities:** Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego.

A completely new city is absent from both development endpoint roles. A role-specific unknown label is absent from its corresponding development column. These measures are kept distinct; they identify the same new cities in the supplied data. Missing categories, if present, are explicit strings rather than dropped rows.

| Final-validation coverage check | Rows | % of all 12,000 rows |
| --- | --- | --- |
| Unseen pickup label | 725 | 6.041667% |
| Unseen delivery label | 722 | 6.016667% |
| Any role-specific unseen city label | 1447 | 12.058333% |
| Completely new city at either endpoint | 1447 | 12.058333% |
| Both endpoints completely new | 0 | 0.000000% |
| Unseen directed route | 1461 | 12.175000% |
| Unseen route with both endpoint labels previously seen in their roles | 14 | 0.116667% |

| New city | Pickup rows | Delivery rows | Any endpoint rows | Supplied (lat, lon) |
| --- | --- | --- | --- | --- |
| Allentown | 90 | 81 | 171 | (40.01684, -73.80889) |
| Charlotte | 88 | 88 | 176 | (34.65272, -81.47202) |
| Chicago | 75 | 104 | 179 | (40.51200, -86.86737) |
| Jackson | 118 | 75 | 193 | (32.07771, -90.92587) |
| Knoxville | 93 | 77 | 170 | (36.25725, -86.02539) |
| Laredo | 89 | 106 | 195 | (25.50000, -97.22681) |
| Norfolk | 88 | 84 | 172 | (37.12278, -75.87106) |
| San Diego | 84 | 107 | 191 | (32.00929, -116.89598) |

Rows involving a completely new city with at least one missing coordinate: **0**. The listed pairs are supplied coordinates, not independently verified geographic centroids.

## Unseen directed routes

There are **736 distinct unseen pickup → delivery routes**, affecting **1,461 rows (12.175%)**. Direction matters: A → B and B → A are distinct. The full list and row counts are in [unseen_routes.csv](unseen_routes.csv).

Most frequent unseen routes:

| Pickup | Delivery | Final rows |
| --- | --- | --- |
| Charleston | Allentown | 7 |
| Jackson | Mobile | 7 |
| Baton Rouge | Chicago | 6 |
| Charleston | Chicago | 6 |
| Charlotte | Baton Rouge | 6 |
| Kansas City | Norfolk | 6 |
| Lexington | Laredo | 6 |
| Phoenix | San Diego | 6 |
| Allentown | Milwaukee | 5 |
| Atlanta | San Diego | 5 |
| Chicago | Hartford | 5 |
| Fresno | Knoxville | 5 |
| Jackson | Oklahoma City | 5 |
| Jackson | San Antonio | 5 |
| Knoxville | Milwaukee | 5 |

Unseen routes can also connect familiar cities. A route lookup alone cannot cover these observations; city labels, coordinates, distance, and operational features remain available.

## Model design and executed prediction checks

The shared feature builder retains all four numeric coordinates and preserves new categorical strings. `src/preprocessing.py` provides an unfitted sklearn processor with median imputation and scaling for numeric features and `OneHotEncoder(handle_unknown="ignore")` for categoricals. Put this processor after `FreightFeatureTransformer` inside the estimator pipeline; fit the entire pipeline on each chronological training fold only. Unknown labels produce zero indicators for that categorical block while numeric coordinates and other features remain available.

CatBoost receives the same engineered DataFrame and explicit string categories. Native inference accepts labels absent from its training vocabulary; it does not require pre-extending categories using final data. Numeric missing values remain NaN for CatBoost.

**Executed diagnostic checks:** fitted a fixed Ridge pipeline and a fixed 12-tree, depth-3 CatBoost regressor on the first seven complete development dates: **1,108 rows (2025-01-01 through 2025-01-07)**. Both predicted all **12,000 final-validation rows** without exceptions, including every new-city and unseen-route row. Each returned **12,000 finite predictions**. All four coordinate columns survived sklearn preprocessing.

These bounded diagnostic fits verify inference compatibility only. They are not production models, were not saved, use no final-validation labels or evaluation set, and produce no submission predictions. No MAE, RMSE, R², or hidden score is available from unlabeled final data. Robustness tests also fit tiny synthetic models to check completely unseen pickup, delivery, route and equipment labels, missing categoricals, and geographic feature retention.

Prediction compatibility does not establish accuracy for new cities. The separate frozen production configuration and development-only selection results are in `docs/model_selection.md`; completed output/scorer checks are recorded in `README.md` and `docs/worklog.md`.

Original scorer, assessment PDF, development/final datasets and blank validation template remain unchanged. The December prediction column may be filled by production; this diagnostic does not modify any supplied input.
