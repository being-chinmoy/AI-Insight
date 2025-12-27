import pandas as pd
import time
import os
import sys
import random
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.prompt import Prompt
from rich.status import Status
from rich.columns import Columns
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Rich Console
console = Console()

class InsightEngine:
    """
    Advanced Analytics Engine 'Training' a live model on the dataset 
    to extract keywords and reasons.
    """
    def __init__(self, df):
        self.df = df
        
    def get_top_keywords(self, text_list, n=3):
        """Extracts top frequent keywords from a list of text."""
        if not text_list:
            return ["No data to analyze."]
            
        try:
            # Simple keyword extraction (Unigrams + Bigrams)
            vec = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10)
            X = vec.fit_transform(text_list)
            sum_words = X.sum(axis=0) 
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
            return [w[0] for w in words_freq[:n]]
        except Exception:
            return ["Insufficient data for text mining."]
            
    def get_positive_highlights(self, subset_df, n=3):
        pos_reviews = subset_df[subset_df['sentiment'] == 'positive']['review_text'].dropna().tolist()
        return self.get_top_keywords(pos_reviews, n)

    def get_negative_reasons(self, subset_df, n=3):
        neg_reviews = subset_df[subset_df['sentiment'] == 'negative']['review_text'].dropna().tolist()
        return self.get_top_keywords(neg_reviews, n)

class ManagerChatSystem:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.regions = []
        self.products = []
        self.platforms = []
        self.genders = []
        self.ages = []
        self.insight_engine = None
        
    def load_data(self):
        """Loads data with a futuristic loading animation."""
        with console.status("[bold green]Establishing Secure Connection to Database...", spinner="dots"):
            time.sleep(1.5) # Fake delay for effect
            try:
                self.df = pd.read_csv(self.data_path)
                # Normalize column names just in case
                self.df.columns = [c.lower().strip() for c in self.df.columns]
                
                # Fix Rating Column Name
                if 'customer_rating' in self.df.columns:
                    self.df.rename(columns={'customer_rating': 'rating'}, inplace=True)
                
                # Fix Platform Column Name
                if 'platform' in self.df.columns:
                    self.df.rename(columns={'platform': 'purchase_platform'}, inplace=True)
                
                # Ensure rating is numeric
                if 'rating' in self.df.columns:
                    self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')

                # Normalize age_group values
                if 'age_group' in self.df.columns:
                    self.df['age_group'] = self.df['age_group'].replace({
                        '50+': '46-60',
                        '36-50': '36-45'
                    })

                # Normalize values to Title Case to avoid duplicates
                for col in ['region', 'product_category', 'purchase_platform', 'gender', 'age_group']:
                    if col in self.df.columns:
                        self.df[col] = self.df[col].astype(str).str.strip().str.title()
                
                self.regions = self.df['region'].unique() if 'region' in self.df.columns else []
                self.products = self.df['product_category'].unique() if 'product_category' in self.df.columns else []
                self.platforms = self.df['purchase_platform'].unique() if 'purchase_platform' in self.df.columns else []
                self.genders = self.df['gender'].unique() if 'gender' in self.df.columns else []
                self.ages = self.df['age_group'].unique() if 'age_group' in self.df.columns else []
                
                self.insight_engine = InsightEngine(self.df)
                
                console.log("[bold cyan]Data Stream Acquired & Insight Engine Trained.[/bold cyan]")
                time.sleep(0.5)
            except FileNotFoundError:
                console.print(Panel("[bold red]CRITICAL ERROR: Data Source Not Found![/bold red]", title="System Alert"))
                sys.exit(1)
            except Exception as e:
                console.print(Panel(f"[bold red]System Failure: {e}[/bold red]", title="System Alert"))
                sys.exit(1)

    def startup_sequence(self):
        """Displays a futuristic startup sequence."""
        console.clear()
        
        # ASCII Art Title
        title = r"""
   __  __                                   
  |  \/  | __ _ _ __   __ _  __ _  ___ _ __ 
  | |\/| |/ _` | '_ \ / _` |/ _` |/ _ \ '__|
  | |  | | (_| | | | | (_| | (_| |  __/ |   
  |_|  |_|\__,_|_| |_|\__,_|\__, |\___|_|   
                            |___/           
        """
        console.print(Panel(Align.center(title, vertical="middle"), style="bold blue", subtitle="v2.4.1 - QUANTUM CORE"))
        time.sleep(1)

        # System Checks
        steps = ["Initializing NLP Core", "Calibrating Sentiment Sensors", "Loading Predictive Models", "Decrypting User Keys"]
        for step in steps:
            console.print(f"[green]>> {step}... OK[/green]")
            time.sleep(0.2)
        
        console.print(Panel("[bold white]Welcome, Manager. System is aligned and ready.[/bold white]", style="bold green"))
        print()

    def analyze_priority_issues(self):
        """Identifies high priority issues."""
        console.print("[bold yellow]Scanning for CRITICAL Priorities...[/bold yellow]")
        
        # Logic: Negative Sentiment + Not Resolved (if exists)
        # Using basic logic for now assuming 'sentiment' and 'issue_resolved' cols
        
        target_df = self.df[self.df['sentiment'] == 'negative']
        if 'issue_resolved' in self.df.columns:
            target_df = target_df[target_df['issue_resolved'] == 'no']
        
        # Top 5 urgent
        top_issues = target_df.head(5)
        
        table = Table(title="PRIORITY LEVEL: HIGH", style="red")
        table.add_column("ID", style="cyan")
        table.add_column("Product", style="magenta")
        table.add_column("Issue", style="white")
        table.add_column("Region", style="green")
        
        for _, row in top_issues.iterrows():
            issue_text = row.get('review_text', 'N/A')[:50] + "..."
            table.add_row(str(row.get('customer_id', '?')), row.get('product_category', 'Unknown'), issue_text, row.get('region', 'Unknown'))
            
        console.print(table)
        console.print(f"[italic red]>> Action Required: {len(target_df)} unresolved critical tickets detected.[/italic red]")

    def analyze_regions(self):
        """Regional performance analysis."""
        if 'region' not in self.df.columns:
            console.print("[red]Region data unavailable.[/red]")
            return

        table = Table(title="REGIONAL DIAGNOSTICS", style="blue")
        table.add_column("Region", style="cyan")
        table.add_column("Total Tickets", style="white")
        table.add_column("Neg. Sentiment %", style="red")
        
        stats = []
        for reg in self.regions:
            reg_df = self.df[self.df['region'] == reg]
            total = len(reg_df)
            neg = len(reg_df[reg_df['sentiment'] == 'negative'])
            neg_pct = (neg / total * 100) if total > 0 else 0
            stats.append((reg, total, neg_pct))
            
        # Sort by worst (highest neg sentiment)
        stats.sort(key=lambda x: x[2], reverse=True)
        
        for s in stats:
            table.add_row(s[0], str(s[1]), f"{s[2]:.1f}%")
            
        console.print(table)
        worst_region = stats[0][0]
        console.print(f"[bold yellow]>> Insight: {worst_region} is showing system instability (Highest Negative Rate). Recommendation: reallocation of support nodes.[/bold yellow]")

    def analyze_demographics(self):
        """Demographics breakdown."""
        console.print(Panel("Demographic Segmentation Analysis", style="bold magenta"))
        
        if 'age_group' in self.df.columns:
            age_counts = self.df['age_group'].value_counts()
            table = Table(title="Age Group Distribution")
            table.add_column("Group", style="magenta")
            table.add_column("Count", style="white")
            for age, count in age_counts.items():
                table.add_row(str(age), str(count))
            console.print(table)
            
    def analyze_performance(self):
        """Operational metrics."""
        if 'response_time_hours' in self.df.columns:
            avg_time = self.df['response_time_hours'].mean()
            console.print(Panel(f"[bold cyan]Average Response Time: {avg_time:.2f} Hours[/bold cyan]", title="Operational Efficiency"))
            
            if avg_time > 24:
                console.print("[bold red]>> WARNING: Response time exceeds SLA (24h). Optimize protocols immediately.[/bold red]")
            else:
                console.print("[bold green]>> Efficiency within optimal parameters.[/bold green]")
        else:
            console.print("[yellow]Response time data missing.[/yellow]")

    def analyze_products(self):
        """Product performance analysis."""
        console.print(Panel("Product Category Analysis", style="bold cyan"))
        if 'product_category' in self.df.columns:
            counts = self.df['product_category'].value_counts()
            table = Table(title="Product Issues Breakdown")
            table.add_column("Category", style="cyan")
            table.add_column("Total Issues", style="white")
            for prod, count in counts.items():
                table.add_row(str(prod), str(count))
            console.print(table)

        if "help" in user_input:
            console.print("[dim]Supported Commands: 'deep dive' (granular), 'platforms' (platform analysis), 'customer <id>', 'priority', 'region', 'products', 'demographics', 'performance', 'exit'[/dim]")
        
    def analyze_platforms(self):
        """Platform performance analysis with advanced insights."""
        console.print(Panel("Purchase Platform Analysis", style="bold cyan"))
        
        if 'purchase_platform' not in self.df.columns:
             console.print("[red]Platform data unavailable.[/red]")
             return

        counts = self.df['purchase_platform'].value_counts()
        
        # Build stats
        platform_stats = []
        for pf in self.platforms:
            pf_df = self.df[self.df['purchase_platform'] == pf]
            if pf_df.empty: continue
            
            avg_r = pf_df['rating'].mean() if 'rating' in pf_df.columns else 0
            issues_df = pf_df[pf_df['sentiment'] == 'negative']
            issues_count = len(issues_df)
            
            # --- Advanced Insights ---
            
            # 1. Product Performance
            best_prod_name = "N/A"
            best_prod_rating = 0
            worst_prod_name = "N/A"
            worst_prod_rating = 5
            most_issue_prod_name = "N/A"
            most_issue_prod_count = 0
            
            if 'product_category' in pf_df.columns:
                # Group by product
                prod_grp = pf_df.groupby('product_category')
                
                # Best/Worst Rating
                if 'rating' in pf_df.columns:
                    prod_ratings = prod_grp['rating'].mean().dropna()
                    if not prod_ratings.empty:
                        best_prod_name = prod_ratings.idxmax()
                        best_prod_rating = prod_ratings.max()
                        worst_prod_name = prod_ratings.idxmin()
                        worst_prod_rating = prod_ratings.min()
                
                # Most Issues
                # Filter original df for this platform + negative
                if not issues_df.empty and 'product_category' in issues_df.columns:
                    issue_counts = issues_df['product_category'].value_counts()
                    if not issue_counts.empty:
                        most_issue_prod_name = issue_counts.idxmax()
                        most_issue_prod_count = issue_counts.max()
            
            # 2. Demographic Hotspots (Age/Gender)
            demo_issue = "N/A"
            if not issues_df.empty:
                # Find most frequent Age Group with issues
                if 'age_group' in issues_df.columns:
                    top_age = issues_df['age_group'].value_counts().idxmax()
                    demo_issue = f"Age {top_age}"
                # Or Gender
                if 'gender' in issues_df.columns:
                    top_gender = issues_df['gender'].value_counts().idxmax()
                    if demo_issue != "N/A":
                        demo_issue += f" / {top_gender}"
                    else:
                        demo_issue = top_gender

            platform_stats.append({
                'Platform': pf,
                'Total': len(pf_df),
                'Rating': avg_r,
                'Issues': issues_count,
                'Best_Prod': f"{best_prod_name} ({best_prod_rating:.1f})",
                'Worst_Prod': f"{worst_prod_name} ({worst_prod_rating:.1f})",
                'Issue_Prod': f"{most_issue_prod_name} ({most_issue_prod_count})",
                'Demo_Risk': demo_issue
            })
        
        # Sort by Rating
        platform_stats.sort(key=lambda x: x['Rating'], reverse=True)
        
        # 1. Overview Table
        table = Table(title="Platform Performance Overview")
        table.add_column("Platform", style="cyan")
        table.add_column("Total", style="white")
        table.add_column("Avg Rating", style="yellow")
        table.add_column("Issues", style="red")
        
        for stat in platform_stats:
            table.add_row(
                stat['Platform'], 
                str(stat['Total']), 
                f"{stat['Rating']:.2f}",
                str(stat['Issues'])
            )
        console.print(table)
        print()
        
        # 2. Detailed Breakdown Grid
        console.rule("[bold magenta]Granular Platform Intelligence[/bold magenta]")
        
        grid = Table.grid(expand=True, padding=(1, 2))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        
        # Create panels for top 4 platforms (to avoid screen overflow if many)
        panels = []
        for stat in platform_stats:
            p_panel = Panel(
                f"""
[bold yellow]Avg Rating:[/bold yellow] {stat['Rating']:.2f}  |  [bold red]Total Issues:[/bold red] {stat['Issues']}

[green]✅ Best Product:[/green] {stat['Best_Prod']}
[red]❌ Worst Product:[/red] {stat['Worst_Prod']}
[red]⚠️ Top Issue Source:[/red] {stat['Issue_Prod']}
[magenta]👥 High Risk Group:[/magenta] {stat['Demo_Risk']}
                """,
                title=f"[bold white]{stat['Platform']}[/bold white]",
                style="cyan",
                border_style="dim"
            )
            panels.append(p_panel)
            
        # Display in columns
        console.print(Columns(panels, expand=True))
        
    def analyze_deep_dive(self):
        """
        Interactive Drill-Down: Region -> Platform -> Age -> Gender -> Insights.
        """
        console.print(Panel("Initiating Multi-Vector Analysis Protocol...", style="bold magenta"))
        
        # 1. Gather Criteria
        criteria = {}

        def get_selection(prompt_text, options):
            """Case-insensitive helper for selection with HINTS."""
            display_options = ['all'] + list(options)
            choices_lower = {opt.lower(): opt for opt in display_options}
            
            # Format choices for display
            hint_str = ", ".join(display_options) if len(display_options) <= 8 else ", ".join(display_options[:8]) + "..."
            
            while True:
                user_input = Prompt.ask(f"[cyan]{prompt_text}[/cyan] (or 'all', {hint_str})", choices=None, default="all")
                if user_input.lower() in choices_lower:
                    return choices_lower[user_input.lower()]
                
                console.print(f"[red]Invalid option. Please choose from: {hint_str}[/red]")

        criteria['region'] = get_selection("Target Region", self.regions)
        
        if len(self.platforms) > 0:
            criteria['purchase_platform'] = get_selection("Target Platform", self.platforms)
            
        if len(self.ages) > 0:
            criteria['age_group'] = get_selection("Target Age Group", self.ages)
            
        if len(self.genders) > 0:
            criteria['gender'] = get_selection("Target Gender", self.genders)

        # 2. Filter Data
        subset = self.df.copy()
        filter_desc = []
        for col, val in criteria.items():
            if val != 'all':
                subset = subset[subset[col] == val]
                filter_desc.append(f"{col}={val}")
            
        if subset.empty:
            console.print("[red]No data found for this specific combination.[/red]")
            return

        desc_str = ", ".join(filter_desc) if filter_desc else "GLOBAL"
        console.rule(f"[bold yellow]Analysis for: {desc_str}[/bold yellow]")

        # 3. Overall Metrics
        total = len(subset)
        avg_rating = subset['rating'].mean() if 'rating' in subset.columns else 0
        avg_resp = subset['response_time_hours'].mean() if 'response_time_hours' in subset.columns else 0
        
        # Grid Layout for Stats
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(
            Panel(f"[bold white]{total}[/bold white]", title="Transactions", style="blue"),
            Panel(f"[bold yellow]{avg_rating:.2f}/5[/bold yellow]", title="Avg Rating", style="yellow"),
            Panel(f"[bold cyan]{avg_resp:.1f} hrs[/bold cyan]", title="Avg Response Time", style="cyan")
        )
        console.print(grid)
        print()

        # 4. NEGATIVE / ISSUES ANALYSIS
        neg_subset = subset[subset['sentiment'] == 'negative']
        neg_count = len(neg_subset)
        if neg_count > 0:
            worst_prod = neg_subset['product_category'].value_counts().idxmax() if 'product_category' in neg_subset.columns else "N/A"
            worst_prod_count = neg_subset['product_category'].value_counts().max() if 'product_category' in neg_subset.columns else 0
            
            neg_reasons = self.insight_engine.get_negative_reasons(neg_subset)
            reasons_str = ", ".join(neg_reasons)
            
            neg_panel = Panel(
                f"""
[bold red]Critical Issues detected: {neg_count}[/bold red]
Most Problematic Sector: [bold white]{worst_prod}[/bold white] ({worst_prod_count} complaints)
Root Causes (AI Detected): [italic red]{reasons_str}[/italic red]
Low Rating (<3) Count: {len(subset[subset['rating'] < 3]) if 'rating' in subset.columns else 'N/A'}
                """,
                title="❌ PROBLEM ANALYSIS",
                style="red",
                border_style="red"
            )
        else:
            neg_panel = Panel("No negative feedback detected.", title="❌ PROBLEM ANALYSIS", style="green")

        # 5. POSITIVE / HIGHLIGHTS ANALYSIS
        pos_subset = subset[subset['sentiment'] == 'positive']
        pos_count = len(pos_subset)
        if pos_count > 0:
            best_prod = pos_subset['product_category'].value_counts().idxmax() if 'product_category' in pos_subset.columns else "N/A"
            pos_highlights = self.insight_engine.get_positive_highlights(pos_subset)
            highlights_str = ", ".join(pos_highlights)
            
            pos_panel = Panel(
                f"""
[bold green]Positive Feedback: {pos_count}[/bold green]
Top Performing Sector: [bold white]{best_prod}[/bold white]
Customer Praises (AI Detected): [italic green]{highlights_str}[/italic green]
                """,
                title="✅ POSITIVE HIGHLIGHTS",
                style="green",
                border_style="green"
            )
        else:
             pos_panel = Panel("No positive feedback detected.", title="✅ POSITIVE HIGHLIGHTS", style="yellow")

        # Display Side-by-Side
        console.print(Columns([neg_panel, pos_panel]))
        print()

        # 6. PLATFORM-WISE BREAKDOWN (If applicable)
        if criteria.get('purchase_platform', 'all') == 'all' and 'purchase_platform' in subset.columns:
            console.rule("[bold cyan]Platform Performance Matrix[/bold cyan]")
            
            # Aggregate Data
            platform_stats = []
            for pf in subset['purchase_platform'].unique():
                pf_df = subset[subset['purchase_platform'] == pf]
                pf_total = len(pf_df)
                if pf_total == 0: continue
                
                avg_r = pf_df['rating'].mean() if 'rating' in pf_df.columns else 0
                issues = len(pf_df[pf_df['sentiment'] == 'negative'])
                platform_stats.append({
                    'Platform': pf,
                    'Rating': avg_r,
                    'Issues': issues,
                    'Total': pf_total
                })
            
            # Sort by Rating (Best first)
            platform_stats.sort(key=lambda x: x['Rating'], reverse=True)
            
            if platform_stats:
                # Table
                p_table = Table(title="Platform Comparison (Ranked by Rating)", expand=True)
                p_table.add_column("Rank", style="dim")
                p_table.add_column("Platform", style="bold cyan")
                p_table.add_column("Avg Rating", justify="right", style="yellow")
                p_table.add_column("Issues", justify="right", style="red")
                p_table.add_column("Status", justify="center")

                for idx, stat in enumerate(platform_stats):
                    rank = str(idx + 1)
                    rating_str = f"{stat['Rating']:.2f}"
                    # Determine status icon
                    if stat['Rating'] >= 4.0: status = "[green]EXCELLENT[/green]"
                    elif stat['Rating'] >= 3.0: status = "[yellow]STABLE[/yellow]"
                    else: status = "[red]CRITICAL[/red]"
                    
                    p_table.add_row(rank, stat['Platform'], rating_str, str(stat['Issues']), status)
                
                console.print(p_table)
                
                # Insights
                best_pf = platform_stats[0]
                worst_pf = min(platform_stats, key=lambda x: x['Rating'])
                most_issues_pf = max(platform_stats, key=lambda x: x['Issues'])
                
                console.print(Panel(
                    f"""
[bold green]🏆 Top Performer:[/bold green] [white]{best_pf['Platform']}[/white] (Rating: {best_pf['Rating']:.2f})
[bold red]⚠️ Needs Attention:[/bold red] [white]{worst_pf['Platform']}[/white] (Lowest Rating: {worst_pf['Rating']:.2f})
[bold yellow]🔥 High Volume Issues:[/bold yellow] [white]{most_issues_pf['Platform']}[/white] ({most_issues_pf['Issues']} tickets)
                    """,
                    title="Platform Strategic Insights",
                    style="blue"
                ))
                print()

    def analyze_specific_customer(self, customer_id):
        """Lookup specific customer."""
        customer_id = str(customer_id).strip()
        row = self.df[self.df['customer_id'].astype(str) == customer_id]
        
        if row.empty:
            console.print(f"[red]Customer ID {customer_id} not found.[/red]")
            return
            
        console.print(Panel(f"profile: Customer #{customer_id}", style="bold green"))
        for col in row.columns:
            val = row.iloc[0][col]
            console.print(f"[cyan]{col}:[/cyan] {val}")
            
    def get_bot_response(self, user_input):
        user_input = user_input.lower()
        
        # Regex-like intent matching
        import re
        
        # Customer Lookup (e.g., "customer 23", "client 90")
        cust_match = re.search(r'(customer|client|id)\s+(no\s+)?(\d+)', user_input)
        if cust_match:
            cid = cust_match.group(3)
            self.analyze_specific_customer(cid)
            return

        if "deep" in user_input or "drill" in user_input or "why" in user_input:
            self.analyze_deep_dive()
        elif "platform" in user_input:
             # Check if user meant deep dive specific to platform or general stats
             if "deep" in user_input or "specific" in user_input:
                 self.analyze_deep_dive()
             else:
                 self.analyze_platforms()
        elif "priority" in user_input or "urgent" in user_input or "high" in user_input:
            self.analyze_priority_issues()
        elif "region" in user_input or "area" in user_input:
            self.analyze_regions()
        elif "product" in user_input or "cat" in user_input:
            self.analyze_products()
        elif "demo" in user_input or "age" in user_input or "gender" in user_input:
             self.analyze_demographics()
        elif "perf" in user_input or "time" in user_input or "sla" in user_input:
             self.analyze_performance()
        elif "help" in user_input:
            console.print("[dim]Supported Commands: 'deep dive' (granular), 'customer <id>', 'priority', 'region', 'products', 'demographics', 'performance', 'exit'[/dim]")
        else:
            # General "Insights" or Fallback
            console.print("[bold white]Accessing Global Dashboard...[/bold white]")
            self.analyze_regions()
            print()
            self.analyze_performance()

    def run(self):
        self.load_data()
        self.startup_sequence()
        
        while True:
            try:
                user_input = console.input("[bold green]Manager_Command > [/bold green]")
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("[bold red]Terminating Session...[/bold red]")
                    break
                
                self.get_bot_response(user_input)
                print() # Spacing
                
            except KeyboardInterrupt:
                console.print("\n[bold red]Forced Disconnect.[/bold red]")
                break

if __name__ == "__main__":
    # Get the directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Customer_Sentiment.csv")
    
    system = ManagerChatSystem(csv_path)
    system.run()
