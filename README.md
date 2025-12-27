# Customer Sentiment "Manager Chat" System

## Overview
This is an advanced, terminal-based AI analytics tool designed to help managers explore customer sentiment data. It features a futuristic "Quantum Core" UI powered by `rich` and provides deep insights into customer feedback, rating trends, and operational issues.

## Prerequisites
- **Python 3.8+**
- **Libraries**: `pandas`, `rich`
- **Data**: Ensure `Customer_Sentiment.csv` is present in the project folder.

## Installation
1.  **Install Dependencies**:
    ```bash
    pip install pandas rich
    ```

## Usage
Run the main script to launch the Quantum Core interface:
```bash
python manager_chat.py
```

## Command Reference

The system accepts natural language-style commands. Here are the most powerful ones:

### 1. `platforms` (or `analyze platforms`)
**[NEW]** Launches the **Platform Intelligence Dashboard**.
-   **Overview**: Summary table of all purchase platforms (Amazon, Flipkart, Myntra, etc.) ranked by performance.
-   **Deep Insights**: For each platform, discover:
    -   ✅ **Best Product Category**
    -   ❌ **Worst Product Category**
    -   ⚠️ **Top Issue Source** (Product with most complaints)
    -   👥 **High Risk Demographic** (Age/Gender most affected)

### 2. `deep dive`
Starts an **Interactive Wizard** to filter data by multiple criteria.
-   **Steps**: You will be prompted to select a Region, Platform, Age Group, and Gender.
-   **Hints**: The prompts now list available options (e.g., `Target Region (or 'all', North, South...)`).
-   **Platform Comparison**: If you select "All" platforms, a comparative matrix is displayed.

### 3. `customer <id>`
Lookup a specific customer profile.
-   **Example**: `customer 12345`
-   Shows their recent transaction, rating, and sentiment analysis of their review.

### 4. `region` / `products` / `demographics`
Quick summary reports for specific dimensions.

### 5. `help`
Displays the list of supported commands.

### 6. `exit`
Terminates the session.

## Data Structure
The system expects `Customer_Sentiment.csv` with columns like:
-   `customer_id`
-   `customer_rating` (automatically renamed to `rating`)
-   `platform` (automatically renamed to `purchase_platform`)
-   `review_text`
-   `region`, `age_group`, `gender`, etc.
