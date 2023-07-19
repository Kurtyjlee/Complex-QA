# Dataset
This dataset includes data that has been scrapped and cleaned. The data can be found under `./data/context/*/*.json`.
* RSIS (2021 and 2023)
* Straitstimes (2023)
* New York times (2021)

<pre> 
📦data
 ┣ 📂context
 ┃ ┣ 📂nyt
 ┃ ┃ ┣ 📜log.json
 ┃ ┃ ┣ 📜nyt_data_2021.json
 ┃ ┃ ┗ 📜nyt_data_2023.json
 ┃ ┣ 📂rsis
 ┃ ┃ ┣ 📂pdf
 ┃ ┃ ┣ 📜data_2021.json
 ┃ ┃ ┗ 📜data_2023.json
 ┃ ┣ 📂straitstimes
</pre>

This dataset also includes generations from the project, under `./data/generations/*/*.json`.

<pre>
 📦data
 ┗ 📂generations
 ┃ ┣ 📂logs  // Error logs
 ┃ ┣ 📂nyt
 ┃ ┃ ┣ 📂QA  // Old QA data
 ┃ ┃ ┣ 📜... // QA data used in the project
 ┃ ┣ 📂old_generations
 ┃ ┣ 📂perplexity
 ┃ ┃ ┣ 📜log.txt  // Error logs
 ┃ ┃ ┣ 📜... // Perplexity data used in the project
 ┃ ┣ 📂rsis
 ┃ ┃ ┣ 📂QA  // Old QA data
 ┃ ┃ ┣ 📂answers  // Old answers generated
 ┃ ┃ ┣ 📂evaluations  // Old evaluations
 ┃ ┃ ┣ 📂questions  // old questions generated
 ┃ ┃ ┣ 📜...  // QA data used in the project
 ┃ ┗ 📂straitstimes
 ┃ ┃ ┣ 📂QA   // Old QA data
 ┃ ┃ ┣ 📜...  // QA data used in the project
</pre>
