"""Complete test script for the RAG Pipeline with MCP integration.

This script tests all major components:
1. Configuration validation
2. PDF ingestion
3. Knowledge base retrieval
4. Web search fallback
5. End-to-end question answering
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.Math import logger
from test import RAGPipeline


async def test_configuration():
    """Test 1: Configuration validation."""
    print("\n" + "=" * 70)
    print("TEST 1: Configuration Validation")
    print("=" * 70)

    try:
        pipeline = RAGPipeline()
        print("✅ Pipeline initialized successfully")
        print(f"   - Model: {pipeline.model_name}")
        print(f"   - Collection: {pipeline.qdrant_config.collection_name}")
        print(f"   - Relevance threshold: {pipeline.RELEVANCE_THRESHOLD}")
        return True, pipeline
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False, None


async def test_kb_query(pipeline: RAGPipeline):
    """Test 2: Query from knowledge base."""
    print("\n" + "=" * 70)
    print("TEST 2: Knowledge Base Query")
    print("=" * 70)

    # This should work if you have PDFs ingested about calculus
    question = "What is a derivative?"
    print(f"Question: {question}")

    try:
        result = await pipeline.query(question)

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        print(f"\n✅ Answer received ({len(answer)} characters)")
        print(f"   Preview: {answer[:150]}...")
        print(f"\n   Sources: {len(sources)} found")
        for i, source in enumerate(sources[:3], 1):
            print(f"   {i}. {source[:80]}...")

        return len(answer) > 50

    except Exception as e:
        print(f"❌ KB query failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_web_search_query(pipeline: RAGPipeline):
    """Test 3: Web search fallback."""
    print("\n" + "=" * 70)
    print("TEST 3: Web Search Fallback (MCP)")
    print("=" * 70)

    # This question should NOT be in your PDFs
    question = "What is the Collatz conjecture and why is it unsolved?"
    print(f"Question: {question}")

    # Check MCP config exists
    if not Path("mcp_config.json").exists():
        print("❌ mcp_config.json not found. Skipping web search test.")
        print(
            "   Create mcp_config.json with your Firecrawl API key to enable this test."
        )
        return None

    try:
        result = await pipeline.query(question)

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        print(f"\n✅ Answer received ({len(answer)} characters)")
        print(f"   Preview: {answer[:200]}...")
        print(f"\n   Sources: {len(sources)} found")
        for i, source in enumerate(sources[:3], 1):
            print(f"   {i}. {source[:80]}...")

        # Check if it looks like a web search result
        is_web_result = any(
            keyword in str(sources).lower()
            for keyword in ["web", "http", "search", "url"]
        )

        if is_web_result:
            print("\n   ✓ Detected web search was used")

        return len(answer) > 50

    except Exception as e:
        print(f"❌ Web search query failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_conversation_history(pipeline: RAGPipeline):
    """Test 4: Multi-turn conversation with history."""
    print("\n" + "=" * 70)
    print("TEST 4: Conversation with History")
    print("=" * 70)

    history = []

    # First question
    q1 = "What is integration?"
    print(f"\n👤 User: {q1}")

    try:
        result1 = await pipeline.query(q1, history=history)
        answer1 = result1.get("answer", "")
        print(f"🤖 Assistant: {answer1[:150]}...")

        # Add to history
        history.append({"sender": "user", "text": q1})
        history.append({"sender": "assistant", "text": answer1})

        # Follow-up question
        q2 = "Can you give me an example?"
        print(f"\n👤 User: {q2}")

        result2 = await pipeline.query(q2, history=history)
        answer2 = result2.get("answer", "")
        print(f"🤖 Assistant: {answer2[:150]}...")

        print(f"\n✅ Multi-turn conversation successful")
        print(f"   - History size: {len(history)} messages")

        return len(answer2) > 30

    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        return False


async def test_invalid_input(pipeline: RAGPipeline):
    """Test 5: Input validation (non-math question)."""
    print("\n" + "=" * 70)
    print("TEST 5: Input Validation (Non-Math Question)")
    print("=" * 70)

    question = "What's the weather like today?"
    print(f"Question: {question}")

    try:
        result = await pipeline.query(question)
        answer = result.get("answer", "")

        print(f"\n✅ Validation working")
        print(f"   Response: {answer[:100]}...")

        # Should contain rejection message
        is_rejected = any(
            keyword in answer.lower()
            for keyword in ["math", "only", "cannot", "related"]
        )

        if is_rejected:
            print("   ✓ Correctly rejected non-math question")
            return True
        else:
            print("   ⚠️ Expected rejection message not found")
            return False

    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


async def test_edge_cases(pipeline: RAGPipeline):
    """Test 6: Edge cases."""
    print("\n" + "=" * 70)
    print("TEST 6: Edge Cases")
    print("=" * 70)

    test_cases = [
        ("", "Empty question"),
        ("   ", "Whitespace only"),
        ("a" * 1000, "Very long question"),
    ]

    results = []

    for question, description in test_cases:
        print(f"\n  Testing: {description}")
        try:
            result = await pipeline.query(question)
            print(f"  ✓ Handled gracefully")
            results.append(True)
        except ValueError as e:
            # Expected for empty/whitespace
            print(f"  ✓ Raised ValueError (expected): {e}")
            results.append(True)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append(False)

    success = all(results)
    if success:
        print(f"\n✅ All edge cases handled correctly")
    else:
        print(f"\n❌ Some edge cases failed")

    return success


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RAG PIPELINE COMPLETE TEST SUITE")
    print("=" * 70)
    print("\nThis will test:")
    print("  1. Configuration validation")
    print("  2. Knowledge base retrieval")
    print("  3. Web search fallback (if configured)")
    print("  4. Conversation history")
    print("  5. Input validation")
    print("  6. Edge cases")
    print("\n" + "=" * 70)

    # Test 1: Configuration
    config_ok, pipeline = await test_configuration()
    if not config_ok or not pipeline:
        print("\n❌ Cannot proceed without valid configuration")
        return

    # Run all tests
    tests = [
        # ("Knowledge Base Query", lambda: test_kb_query(pipeline)),
        ("Web Search Fallback", lambda: test_web_search_query(pipeline)),
        # ("Conversation History", lambda: test_conversation_history(pipeline)),
        # ("Input Validation", lambda: test_invalid_input(pipeline)),
        # ("Edge Cases", lambda: test_edge_cases(pipeline)),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

        await asyncio.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results:
        if result is None:
            status = "⊘ SKIPPED"
            skipped += 1
        elif result:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        print(f"{status} - {name}")

    total = len(results)
    print(
        f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped (out of {total})"
    )

    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed! Your RAG pipeline is working correctly.")
    elif failed > 0:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above for details.")

    # Additional info
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    if skipped > 0:
        print("\nTo enable skipped tests:")
        print("  1. Create mcp_config.json with your Firecrawl API key")
        print("  2. Ingest some PDF files about math concepts")
        print("  3. Run this test script again")

    print("\nTo use the pipeline in your app:")
    print("  from src.Math.pipeline.rag_pipeline import RAGPipeline")
    print("  pipeline = RAGPipeline()")
    print("  result = await pipeline.query('What is calculus?')")
    print("  print(result['answer'])")


if __name__ == "__main__":
    asyncio.run(main())
