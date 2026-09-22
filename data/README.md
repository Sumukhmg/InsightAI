# InsightAI Data Directory

This folder stores datasets used by InsightAI.

### Default Dataset: UCI Online Retail II
- **Source**: [UCI Machine Learning Repository: Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Direct Download**: `https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx`
- **Description**: Contains approximately 1,067,371 transactions across two years (01/12/2009 to 09/12/2011) for a UK-based non-store online retail business.
- **Sheets**:
  - `Year 2009-2010` (approx. 525,461 rows)
  - `Year 2010-2011` (approx. 541,910 rows)

### Automatic Download & Caching
When running InsightAI, the application provides an automated 1-click download tool. Once downloaded, the raw Excel file is parsed and saved as a high-performance Apache Parquet file (`data/online_retail_II.parquet`), allowing near-instantaneous load times on subsequent launches.

### Custom Uploads
Users can also upload custom CSV or Excel files with compatible schema (`Invoice`/`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`/`UnitPrice`, `Customer ID`, `Country`).
