"""
Association Rule Mining Module
Discovers co-resistance patterns using Apriori and FP-Growth algorithms
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResistanceAssociationMiner:
    """Mine association rules for co-resistance patterns"""
    
    def __init__(self):
        """Initialize association rule miner"""
        self.data = None
        self.binary_data = None
        self.frequent_itemsets_apriori = None
        self.frequent_itemsets_fpgrowth = None
        self.rules_apriori = None
        self.rules_fpgrowth = None
    
    def load_data(self, data_path):
        """Load feature-engineered data"""
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} samples")
        return self
    
    def prepare_binary_matrix(self):
        """Prepare binary resistance matrix for association mining"""
        logger.info("Preparing binary resistance matrix...")
        
        # Get resistance columns
        resistance_cols = [col for col in self.data.columns if col.endswith('_resistant')]
        
        if not resistance_cols:
            raise ValueError("No resistance columns found in data")
        
        # Create binary matrix (1 = resistant, 0 = not resistant)
        self.binary_data = self.data[resistance_cols].copy()
        
        # Rename columns to antibiotic names
        self.binary_data.columns = [col.replace('_resistant', '') for col in resistance_cols]
        
        # Add additional categorical features
        if 'esbl_positive' in self.data.columns:
            self.binary_data['esbl_positive'] = self.data['esbl_positive']
        
        # Add MAR class
        if 'mar_class' in self.data.columns:
            # Create binary columns for each MAR class
            for mar_class in ['low', 'medium', 'high']:
                self.binary_data[f'mar_{mar_class}'] = (self.data['mar_class'] == mar_class).astype(int)
        
        # Add MDR category
        if 'mdr_category' in self.data.columns:
            for mdr_cat in ['non-mdr', 'mdr', 'high-mdr']:
                self.binary_data[f'mdr_{mdr_cat}'] = (self.data['mdr_category'] == mdr_cat).astype(int)
        
        # Ensure all values are 0 or 1
        self.binary_data = self.binary_data.astype(int)
        
        logger.info(f"Binary matrix prepared: {self.binary_data.shape[0]} samples x {self.binary_data.shape[1]} features")
        return self
    
    def mine_apriori(self, min_support=0.1, min_confidence=0.6, min_lift=1.0):
        """
        Mine frequent itemsets and association rules using Apriori
        
        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
        """
        logger.info(f"Mining with Apriori (support={min_support}, confidence={min_confidence}, lift={min_lift})...")
        
        # Find frequent itemsets
        self.frequent_itemsets_apriori = apriori(
            self.binary_data,
            min_support=min_support,
            use_colnames=True
        )
        
        logger.info(f"Found {len(self.frequent_itemsets_apriori)} frequent itemsets")
        
        if len(self.frequent_itemsets_apriori) == 0:
            logger.warning("No frequent itemsets found. Try lowering min_support.")
            return self
        
        # Generate association rules
        self.rules_apriori = association_rules(
            self.frequent_itemsets_apriori,
            metric="confidence",
            min_threshold=min_confidence
        )
        
        # Filter by lift
        self.rules_apriori = self.rules_apriori[self.rules_apriori['lift'] >= min_lift]
        
        # Sort by confidence and lift
        self.rules_apriori = self.rules_apriori.sort_values(['confidence', 'lift'], ascending=False)
        
        logger.info(f"Generated {len(self.rules_apriori)} association rules")
        
        return self
    
    def mine_fpgrowth(self, min_support=0.1, min_confidence=0.6, min_lift=1.0):
        """
        Mine frequent itemsets and association rules using FP-Growth
        
        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
        """
        logger.info(f"Mining with FP-Growth (support={min_support}, confidence={min_confidence}, lift={min_lift})...")
        
        # Find frequent itemsets
        self.frequent_itemsets_fpgrowth = fpgrowth(
            self.binary_data,
            min_support=min_support,
            use_colnames=True
        )
        
        logger.info(f"Found {len(self.frequent_itemsets_fpgrowth)} frequent itemsets")
        
        if len(self.frequent_itemsets_fpgrowth) == 0:
            logger.warning("No frequent itemsets found. Try lowering min_support.")
            return self
        
        # Generate association rules
        self.rules_fpgrowth = association_rules(
            self.frequent_itemsets_fpgrowth,
            metric="confidence",
            min_threshold=min_confidence
        )
        
        # Filter by lift
        self.rules_fpgrowth = self.rules_fpgrowth[self.rules_fpgrowth['lift'] >= min_lift]
        
        # Sort by confidence and lift
        self.rules_fpgrowth = self.rules_fpgrowth.sort_values(['confidence', 'lift'], ascending=False)
        
        logger.info(f"Generated {len(self.rules_fpgrowth)} association rules")
        
        return self
    
    def format_rule(self, antecedents, consequents, confidence, support, lift):
        """Format a rule for display"""
        ant_str = ", ".join(list(antecedents))
        cons_str = ", ".join(list(consequents))
        return f"If [{ant_str}] → Then [{cons_str}] (Conf: {confidence:.2%}, Supp: {support:.2%}, Lift: {lift:.2f})"
    
    def print_top_rules(self, rules_df, method_name, top_n=20):
        """Print top association rules"""
        if rules_df is None or len(rules_df) == 0:
            logger.warning(f"No rules found for {method_name}")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP {top_n} CO-RESISTANCE RULES ({method_name})")
        logger.info(f"{'='*80}")
        
        for idx, row in rules_df.head(top_n).iterrows():
            rule_str = self.format_rule(
                row['antecedents'],
                row['consequents'],
                row['confidence'],
                row['support'],
                row['lift']
            )
            logger.info(f"\n{idx+1}. {rule_str}")
    
    def analyze_rules(self):
        """Analyze and categorize discovered rules"""
        logger.info("\nAnalyzing association rules...")
        
        for method_name, rules_df in [('Apriori', self.rules_apriori), ('FP-Growth', self.rules_fpgrowth)]:
            if rules_df is None or len(rules_df) == 0:
                continue
            
            logger.info(f"\n=== {method_name} Rule Analysis ===")
            
            # Statistics
            logger.info(f"Total rules: {len(rules_df)}")
            logger.info(f"Average confidence: {rules_df['confidence'].mean():.2%}")
            logger.info(f"Average support: {rules_df['support'].mean():.2%}")
            logger.info(f"Average lift: {rules_df['lift'].mean():.2f}")
            
            # High confidence rules (>80%)
            high_conf_rules = rules_df[rules_df['confidence'] > 0.8]
            logger.info(f"High confidence rules (>80%): {len(high_conf_rules)}")
            
            # Strong association rules (lift > 2)
            strong_rules = rules_df[rules_df['lift'] > 2]
            logger.info(f"Strong association rules (lift > 2): {len(strong_rules)}")
            
            # Print top rules
            self.print_top_rules(rules_df, method_name, top_n=15)
        
        return self
    
    def identify_esbl_patterns(self):
        """Identify ESBL-related resistance patterns"""
        logger.info("\n=== ESBL-Related Patterns ===")
        
        for method_name, rules_df in [('Apriori', self.rules_apriori), ('FP-Growth', self.rules_fpgrowth)]:
            if rules_df is None or len(rules_df) == 0:
                continue
            
            # Find rules with ESBL in antecedents
            esbl_rules = rules_df[rules_df['antecedents'].apply(
                lambda x: 'esbl_positive' in x
            )]
            
            if len(esbl_rules) > 0:
                logger.info(f"\n{method_name} - ESBL patterns found: {len(esbl_rules)}")
                self.print_top_rules(esbl_rules, f"{method_name} ESBL", top_n=10)
        
        return self
    
    def identify_mar_patterns(self):
        """Identify MAR-related resistance patterns"""
        logger.info("\n=== MAR-Related Patterns ===")
        
        for method_name, rules_df in [('Apriori', self.rules_apriori), ('FP-Growth', self.rules_fpgrowth)]:
            if rules_df is None or len(rules_df) == 0:
                continue
            
            # Find rules with MAR in antecedents
            mar_rules = rules_df[rules_df['antecedents'].apply(
                lambda x: any('mar_' in str(item) for item in x)
            )]
            
            if len(mar_rules) > 0:
                logger.info(f"\n{method_name} - MAR patterns found: {len(mar_rules)}")
                self.print_top_rules(mar_rules, f"{method_name} MAR", top_n=10)
        
        return self
    
    def identify_multi_drug_patterns(self):
        """Identify patterns involving multiple antibiotics"""
        logger.info("\n=== Multi-Drug Resistance Patterns ===")
        
        for method_name, rules_df in [('Apriori', self.rules_apriori), ('FP-Growth', self.rules_fpgrowth)]:
            if rules_df is None or len(rules_df) == 0:
                continue
            
            # Find rules with 3+ antibiotics in antecedents
            multi_drug_rules = rules_df[rules_df['antecedents'].apply(lambda x: len(x) >= 3)]
            
            if len(multi_drug_rules) > 0:
                logger.info(f"\n{method_name} - Multi-drug patterns found: {len(multi_drug_rules)}")
                self.print_top_rules(multi_drug_rules, f"{method_name} Multi-Drug", top_n=10)
        
        return self
    
    def save_results(self, output_dir):
        """Save association rules and frequent itemsets"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving association mining results to {output_dir}")
        
        # Save Apriori results
        if self.frequent_itemsets_apriori is not None:
            self.frequent_itemsets_apriori.to_csv(output_dir / 'apriori_frequent_itemsets.csv', index=False)
        
        if self.rules_apriori is not None and len(self.rules_apriori) > 0:
            # Convert frozensets to strings for CSV
            rules_df = self.rules_apriori.copy()
            rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
            rules_df.to_csv(output_dir / 'apriori_rules.csv', index=False)
        
        # Save FP-Growth results
        if self.frequent_itemsets_fpgrowth is not None:
            self.frequent_itemsets_fpgrowth.to_csv(output_dir / 'fpgrowth_frequent_itemsets.csv', index=False)
        
        if self.rules_fpgrowth is not None and len(self.rules_fpgrowth) > 0:
            rules_df = self.rules_fpgrowth.copy()
            rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
            rules_df.to_csv(output_dir / 'fpgrowth_rules.csv', index=False)
        
        # Save binary matrix
        self.binary_data.to_csv(output_dir / 'binary_resistance_matrix.csv', index=False)
        
        logger.info("Association mining results saved successfully")
        return self


def main():
    """Main execution function"""
    # Paths
    base_path = Path(__file__).parent.parent.parent
    features_path = base_path / 'data' / 'processed' / 'features.csv'
    results_dir = base_path / 'data' / 'results' / 'association_rules'
    
    # Mine association rules
    miner = ResistanceAssociationMiner()
    (miner.load_data(features_path)
          .prepare_binary_matrix()
          .mine_apriori(min_support=0.1, min_confidence=0.6, min_lift=1.2)
          .mine_fpgrowth(min_support=0.1, min_confidence=0.6, min_lift=1.2)
          .analyze_rules()
          .identify_esbl_patterns()
          .identify_mar_patterns()
          .identify_multi_drug_patterns()
          .save_results(results_dir))
    
    logger.info("\n✓ Association rule mining completed successfully!")


if __name__ == "__main__":
    main()
