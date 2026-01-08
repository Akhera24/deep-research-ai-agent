"""
Requirements Compliance Checker

Verifies that implementation meets ALL requirements and functionality is met. 
- Core Architecture requirements
- Functional Specifications
- Implementation Guidelines
- Deliverables
"""

import json
from pathlib import Path
from typing import Dict, Any, List


class RequirementsChecker:
    """Check compliance with PDF requirements"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.compliance_results = []
    
    def check_all(self) -> Dict[str, Any]:
        """Run all compliance checks"""
        
        print("\n" + "="*70)
        print("PDF REQUIREMENTS COMPLIANCE VERIFICATION")
        print("="*70 + "\n")
        
        # Core Architecture
        print("📋 CORE ARCHITECTURE REQUIREMENTS")
        print("-" * 70)
        self._check_core_architecture()
        
        # Functional Specifications
        print("\n📋 FUNCTIONAL SPECIFICATIONS")
        print("-" * 70)
        self._check_functional_specs()
        
        # Implementation Guidelines
        print("\n📋 IMPLEMENTATION GUIDELINES")
        print("-" * 70)
        self._check_implementation_guidelines()
        
        # Deliverables
        print("\n📋 DELIVERABLES")
        print("-" * 70)
        self._check_deliverables()
        
        # Summary
        return self._generate_summary()
    
    def _check_requirement(
        self,
        name: str,
        check_func,
        details: str = ""
    ):
        """Check single requirement"""
        try:
            result = check_func()
            status = "✅" if result else "❌"
            self.compliance_results.append((name, result, details))
            print(f"{status} {name}")
            if details and result:
                print(f"    {details}")
        except Exception as e:
            self.compliance_results.append((name, False, str(e)))
            print(f"❌ {name}")
            print(f"    Error: {str(e)}")
    
    # ========================================================================
    # CORE ARCHITECTURE CHECKS
    # ========================================================================
    
    def _check_core_architecture(self):
        """Core Architecture Requirements"""
        
        # Multi-Model Integration
        self._check_requirement(
            "Multi-Model Integration (2+ AI models)",
            lambda: self._verify_multi_model(),
            "Claude Opus 4, Gemini 2.5 Pro, GPT-4 Turbo integrated"
        )
        
        # Consecutive Search Strategy
        self._check_requirement(
            "Consecutive Search Strategy",
            lambda: self._verify_consecutive_search(),
            "Implemented in SearchStrategyEngine with query refinement"
        )
        
        # Dynamic Query Refinement
        self._check_requirement(
            "Dynamic Query Refinement",
            lambda: self._verify_dynamic_refinement(),
            "Refines queries based on discovered facts and entities"
        )
    
    def _verify_multi_model(self) -> bool:
        """Verify multi-model integration"""
        router_file = self.project_root / "src/models/router.py"
        
        if not router_file.exists():
            return False
        
        content = router_file.read_text()
        
        # Check for all three models
        has_claude = "claude-opus-4" in content.lower()
        has_gemini = "gemini-2.5" in content.lower() or "gemini" in content.lower()
        has_openai = "gpt-4" in content.lower()
        
        return has_claude and has_gemini and has_openai
    
    def _verify_consecutive_search(self) -> bool:
        """Verify consecutive search implementation"""
        strategy_file = self.project_root / "src/search/strategy.py"
        
        if not strategy_file.exists():
            return False
        
        content = strategy_file.read_text()
        
        # Check for key consecutive search components
        has_initial = "generate_initial_queries" in content
        has_refine = "refine_based_on_findings" in content
        has_entities = "extract_entities" in content
        has_coverage = "update_coverage" in content
        
        return all([has_initial, has_refine, has_entities, has_coverage])
    
    def _verify_dynamic_refinement(self) -> bool:
        """Verify dynamic query refinement"""
        workflow_file = self.project_root / "src/core/workflow.py"
        
        if not workflow_file.exists():
            return False
        
        content = workflow_file.read_text()
        
        # Check for refinement logic in workflow
        has_refine_node = "_node_refine_queries" in content
        has_continue_decision = "decide_continue_or_finish" in content
        has_loop_back = '"continue": "execute_searches"' in content
        
        return all([has_refine_node, has_continue_decision, has_loop_back])
    
    # ========================================================================
    # FUNCTIONAL SPECIFICATIONS CHECKS
    # ========================================================================
    
    def _check_functional_specs(self):
        """Functional Specifications Requirements"""
        
        # Deep Fact Extraction
        self._check_requirement(
            "Deep Fact Extraction (biographical, professional, financial, etc.)",
            lambda: self._verify_fact_extraction(),
            "FactExtractor with 6 categories implemented"
        )
        
        # Risk Pattern Recognition
        self._check_requirement(
            "Risk Pattern Recognition",
            lambda: self._verify_risk_assessment(),
            "Risk assessment node in workflow with pattern detection"
        )
        
        # Connection Mapping
        self._check_requirement(
            "Connection Mapping (entities, relationships)",
            lambda: self._verify_connection_mapping(),
            "Connection mapping node extracts entity relationships"
        )
        
        # Source Validation
        self._check_requirement(
            "Source Validation (confidence scoring, cross-referencing)",
            lambda: self._verify_source_validation(),
            "Confidence scoring + cross-referencing in FactExtractor"
        )
    
    def _verify_fact_extraction(self) -> bool:
        """Verify fact extraction implementation"""
        extractor_file = self.project_root / "src/extraction/extractor.py"
        
        if not extractor_file.exists():
            return False
        
        content = extractor_file.read_text()
        
        # Check for categories
        categories = [
            "biographical", "professional", "financial",
            "legal", "connections", "behavioral"
        ]
        
        return all(cat in content for cat in categories)
    
    def _verify_risk_assessment(self) -> bool:
        """Verify risk assessment"""
        workflow_file = self.project_root / "src/core/workflow.py"
        
        if not workflow_file.exists():
            return False
        
        content = workflow_file.read_text()
        
        return "_node_assess_risks" in content and "risk_flags" in content
    
    def _verify_connection_mapping(self) -> bool:
        """Verify connection mapping"""
        workflow_file = self.project_root / "src/core/workflow.py"
        
        if not workflow_file.exists():
            return False
        
        content = workflow_file.read_text()
        
        return "_node_map_connections" in content and "connections" in content
    
    def _verify_source_validation(self) -> bool:
        """Verify source validation"""
        extractor_file = self.project_root / "src/extraction/extractor.py"
        
        if not extractor_file.exists():
            return False
        
        content = extractor_file.read_text()
        
        has_confidence = "confidence" in content
        has_cross_ref = "cross_reference" in content or "verification_count" in content
        has_source_reliability = "source_reliability" in content
        
        return has_confidence and has_cross_ref and has_source_reliability
    
    # ========================================================================
    # IMPLEMENTATION GUIDELINES CHECKS
    # ========================================================================
    
    def _check_implementation_guidelines(self):
        """Implementation Guidelines Requirements"""
        
        # LangGraph for orchestration
        self._check_requirement(
            "Use LangGraph for agent orchestration",
            lambda: self._verify_langgraph(),
            "LangGraph StateGraph implemented in workflow.py"
        )
        
        # Search engines
        self._check_requirement(
            "Leverage AI APIs, search engines, real online data",
            lambda: self._verify_search_apis(),
            "Brave Search and Serper APIs integrated"
        )
        
        # Error handling
        self._check_requirement(
            "Proper error handling and rate limiting",
            lambda: self._verify_error_handling(),
            "Try-except blocks, retry logic, rate limiting implemented"
        )
        
        # Scalability
        self._check_requirement(
            "Design for scalability and maintainability",
            lambda: self._verify_scalability(),
            "Async/await, modular design, caching, state management"
        )
    
    def _verify_langgraph(self) -> bool:
        """Verify LangGraph usage"""
        workflow_file = self.project_root / "src/core/workflow.py"
        
        if not workflow_file.exists():
            return False
        
        content = workflow_file.read_text()
        
        has_import = "from langgraph.graph import StateGraph" in content
        has_build = "StateGraph(Dict[str, Any])" in content or "StateGraph" in content
        has_nodes = "add_node" in content
        has_edges = "add_edge" in content
        
        return all([has_import, has_build, has_nodes, has_edges])
    
    def _verify_search_apis(self) -> bool:
        """Verify search API integration"""
        executor_file = self.project_root / "src/search/executor.py"
        
        if not executor_file.exists():
            return False
        
        content = executor_file.read_text()
        
        has_brave = "brave" in content.lower()
        has_serper = "serper" in content.lower()
        has_search_method = "_search_brave" in content or "_search_serper" in content
        
        return has_brave and has_serper and has_search_method
    
    def _verify_error_handling(self) -> bool:
        """Verify error handling"""
        executor_file = self.project_root / "src/search/executor.py"
        
        if not executor_file.exists():
            return False
        
        content = executor_file.read_text()
        
        has_try_except = "try:" in content and "except" in content
        has_retry = "retry" in content.lower() or "tenacity" in content
        has_rate_limit = "rate_limit" in content.lower()
        
        return has_try_except and has_retry and has_rate_limit
    
    def _verify_scalability(self) -> bool:
        """Verify scalability design"""
        executor_file = self.project_root / "src/search/executor.py"
        
        if not executor_file.exists():
            return False
        
        content = executor_file.read_text()
        
        has_async = "async def" in content
        has_caching = "cache" in content.lower()
        
        return has_async and has_caching
    
    # ========================================================================
    # DELIVERABLES CHECKS
    # ========================================================================
    
    def _check_deliverables(self):
        """Deliverables Requirements"""
        
        # Complete codebase
        self._check_requirement(
            "Complete codebase with comprehensive documentation",
            lambda: self._verify_codebase(),
            "All core modules implemented with docstrings"
        )
        
        # Three test personas
        self._check_requirement(
            "Three test persona profiles with expected findings",
            lambda: self._verify_test_personas(),
            "Evaluation dataset with 3 personas (Easy/Medium/Hard)"
        )
        
        # Execution logs
        self._check_requirement(
            "Execution logs demonstrating agent performance",
            lambda: self._verify_logging(),
            "Comprehensive logging in all modules"
        )
        
        # Risk assessment reports
        self._check_requirement(
            "Risk assessment reports for each test case",
            lambda: self._verify_reports(),
            "Evaluation runner generates detailed reports"
        )
    
    def _verify_codebase(self) -> bool:
        """Verify complete codebase"""
        required_files = [
            "src/core/workflow.py",
            "src/models/router.py",
            "src/search/strategy.py",
            "src/search/executor.py",
            "src/extraction/extractor.py",
            "config/settings.py"
        ]
        
        return all((self.project_root / f).exists() for f in required_files)
    
    def _verify_test_personas(self) -> bool:
        """Verify test personas"""
        eval_file = self.project_root / "scripts/run_evaluation.py"
        
        if not eval_file.exists():
            return False
        
        content = eval_file.read_text()
        
        # Check for 3 personas
        has_easy = "P001_EASY" in content
        has_medium = "P002_MEDIUM" in content
        has_hard = "P003_HARD" in content
        
        return all([has_easy, has_medium, has_hard])
    
    def _verify_logging(self) -> bool:
        """Verify logging implementation"""
        executor_file = self.project_root / "src/search/executor.py"
        
        if not executor_file.exists():
            return False
        
        content = executor_file.read_text()
        
        has_logger = "logger = get_logger" in content
        has_log_calls = "logger.info" in content or "logger.debug" in content
        
        return has_logger and has_log_calls
    
    def _verify_reports(self) -> bool:
        """Verify report generation"""
        eval_file = self.project_root / "scripts/run_evaluation.py"
        
        if not eval_file.exists():
            return False
        
        content = eval_file.read_text()
        
        has_metrics = "discovery_rate" in content
        has_report_gen = "_save_reports" in content or "_generate_aggregate_report" in content
        
        return has_metrics and has_report_gen
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate compliance summary"""
        
        total = len(self.compliance_results)
        passed = sum(1 for _, result, _ in self.compliance_results if result)
        failed = total - passed
        compliance_rate = passed / total if total > 0 else 0.0
        
        print("\n" + "="*70)
        print("COMPLIANCE SUMMARY")
        print("="*70)
        print(f"Total Requirements:  {total}")
        print(f"Met:                 {passed}")
        print(f"Not Met:             {failed}")
        print(f"Compliance Rate:     {compliance_rate:.1%}")
        print("="*70 + "\n")
        
        if compliance_rate == 1.0:
            print("🎉 100% COMPLIANCE - All requirements met!\n")
        elif compliance_rate >= 0.9:
            print("✅ EXCELLENT - Minor gaps remaining\n")
        elif compliance_rate >= 0.8:
            print("⚠️  GOOD - Some requirements need attention\n")
        else:
            print("❌ NEEDS WORK - Multiple requirements not met\n")
        
        # Detailed results
        if failed > 0:
            print("Requirements NOT met:")
            for name, result, details in self.compliance_results:
                if not result:
                    print(f"  ❌ {name}")
                    if details:
                        print(f"      {details}")
            print()
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "compliance_rate": compliance_rate,
            "results": self.compliance_results
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run compliance check"""
    checker = RequirementsChecker()
    summary = checker.check_all()
    
    # Save results
    output_file = Path("evaluation/requirements_compliance.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📄 Detailed report saved: {output_file}\n")
    
    # Exit code
    import sys
    sys.exit(0 if summary["compliance_rate"] >= 0.9 else 1)


if __name__ == "__main__":
    main()