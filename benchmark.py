#!/usr/bin/env python3
"""
ChatterBox TTS API Benchmark Script

This script tests the performance and scalability of the ChatterBox TTS API
by sending concurrent requests with varying text lengths and measuring
key performance metrics.
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import random
import sys
from pathlib import Path
import csv

@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    concurrent_requests: int
    text_length: int
    text_content: str
    start_time: float
    first_chunk_time: Optional[float]
    end_time: float
    total_time: float
    time_to_first_chunk: Optional[float]
    success: bool
    error_message: Optional[str]
    audio_size_bytes: int
    status_code: Optional[int]
    chunks_received: int

@dataclass
class ConcurrencyMetrics:
    """Aggregated metrics for a concurrency level"""
    concurrent_requests: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_total_time: float
    avg_time_to_first_chunk: float
    min_total_time: float
    max_total_time: float
    median_total_time: float
    p95_total_time: float
    avg_audio_size: float
    requests_per_second: float
    error_rate: float
    avg_text_length: int

class ChatterBoxBenchmark:
    def __init__(self, base_url: str = "http://localhost:6401"):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/stream/audio/speech"
        
        # Test texts with different lengths
        self.test_texts = [
            # Short texts (10-30 words)
            "Hello world, this is a simple test.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing the ChatterBox text-to-speech system with short sentences.",
            
            # Medium texts (50-100 words)
            "This is a medium-length text designed to test the performance of the text-to-speech system under moderate load conditions. The system should handle this reasonably well while maintaining good audio quality and reasonable response times.",
            "Artificial intelligence has revolutionized many aspects of our daily lives, from voice assistants to automated customer service systems. Text-to-speech technology plays a crucial role in making information accessible to everyone, including those with visual impairments or reading difficulties.",
            "The weather today is absolutely beautiful with clear blue skies and gentle breezes. It's the perfect day for outdoor activities like hiking, picnicking, or simply taking a leisurely walk through the park while enjoying the fresh air and sunshine.",
            
            # Long texts (150-250 words)
            "In the rapidly evolving landscape of artificial intelligence and machine learning, text-to-speech synthesis has emerged as one of the most practical and widely adopted applications. Modern TTS systems leverage deep neural networks, particularly transformer architectures, to generate highly natural and expressive speech that closely mimics human intonation, rhythm, and emotional expression. These systems have found applications across numerous domains, from accessibility tools for visually impaired users to voice assistants, audiobook narration, and interactive voice response systems in customer service environments.",
            "The development of multilingual text-to-speech systems presents unique challenges and opportunities in the field of natural language processing. Unlike monolingual systems that focus on perfecting speech synthesis for a single language, multilingual TTS must handle the phonetic, prosodic, and cultural variations across different languages while maintaining consistent quality and naturalness. This requires sophisticated models that can understand and generate the subtle nuances of pronunciation, stress patterns, and intonation that characterize each language, making it a complex but rewarding area of research and development.",
            "Climate change represents one of the most pressing challenges of our time, affecting ecosystems, weather patterns, and human societies across the globe. The scientific consensus clearly indicates that human activities, particularly the emission of greenhouse gases from fossil fuel combustion, deforestation, and industrial processes, are the primary drivers of current climate change. Rising global temperatures have led to melting ice caps, rising sea levels, more frequent extreme weather events, and shifts in precipitation patterns that threaten agriculture, water security, and biodiversity worldwide.",
            
            # Very long texts (300+ words)
            "The history of human civilization is marked by remarkable innovations that have transformed how we live, work, and communicate with one another. From the invention of the wheel and the development of written language to the industrial revolution and the digital age, each breakthrough has built upon previous discoveries to create increasingly sophisticated societies. The printing press democratized knowledge, the steam engine powered industrialization, electricity illuminated our world, and the internet connected us globally in ways our ancestors could never have imagined. Today, we stand at the threshold of another revolutionary period, where artificial intelligence, quantum computing, biotechnology, and renewable energy technologies promise to reshape our future in profound and unprecedented ways. These emerging technologies offer the potential to solve some of humanity's greatest challenges, from climate change and disease to poverty and inequality, while also raising important questions about ethics, privacy, employment, and the very nature of human experience in an increasingly automated world. As we navigate this complex landscape of technological advancement, it becomes crucial to approach innovation with wisdom, ensuring that the benefits are shared broadly and that we preserve the values and principles that define our humanity."
        ]
        
    async def make_request(self, session: aiohttp.ClientSession, text: str, concurrent_count: int) -> RequestMetrics:
        """Make a single request and measure metrics"""
        payload = {
            "input": text,
            "language_id": "en",
            "speed": 1.0,
            "exaggeration": 0.7,
            "cfg_weight": 0.3,
            "temperature": 0.8,
            "chunk_size": 25,
            "output_sample_rate": 8000
        }
        
        metrics = RequestMetrics(
            concurrent_requests=concurrent_count,
            text_length=len(text.split()),
            text_content=text[:100] + "..." if len(text) > 100 else text,
            start_time=time.time(),
            first_chunk_time=None,
            end_time=0,
            total_time=0,
            time_to_first_chunk=None,
            success=False,
            error_message=None,
            audio_size_bytes=0,
            status_code=None,
            chunks_received=0
        )
        
        try:
            async with session.post(self.endpoint, json=payload) as response:
                metrics.status_code = response.status
                
                if response.status != 200:
                    metrics.error_message = f"HTTP {response.status}: {await response.text()}"
                    metrics.end_time = time.time()
                    metrics.total_time = metrics.end_time - metrics.start_time
                    return metrics
                
                # Read streaming response
                first_chunk = True
                async for chunk in response.content.iter_chunked(1024):
                    if first_chunk:
                        metrics.first_chunk_time = time.time()
                        metrics.time_to_first_chunk = metrics.first_chunk_time - metrics.start_time
                        first_chunk = False
                    
                    metrics.audio_size_bytes += len(chunk)
                    metrics.chunks_received += 1
                
                metrics.success = True
                
        except asyncio.TimeoutError:
            metrics.error_message = "Request timeout"
        except aiohttp.ClientError as e:
            metrics.error_message = f"Client error: {str(e)}"
        except Exception as e:
            metrics.error_message = f"Unexpected error: {str(e)}"
        finally:
            metrics.end_time = time.time()
            metrics.total_time = metrics.end_time - metrics.start_time
            
        return metrics
    
    async def run_concurrency_test(self, concurrent_requests: int, requests_per_level: int = 10) -> List[RequestMetrics]:
        """Run benchmark for a specific concurrency level"""
        print(f"\n{'='*60}")
        print(f"Testing {concurrent_requests} concurrent request{'s' if concurrent_requests > 1 else ''}")
        print(f"{'='*60}")
        
        # Create random selection of texts for this test
        selected_texts = random.choices(self.test_texts, k=requests_per_level)
        
        timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = []
            
            # Create batches of concurrent requests
            for i in range(0, requests_per_level, concurrent_requests):
                batch_size = min(concurrent_requests, requests_per_level - i)
                batch_texts = selected_texts[i:i + batch_size]
                
                # Create tasks for this batch
                batch_tasks = [
                    self.make_request(session, text, concurrent_requests)
                    for text in batch_texts
                ]
                
                print(f"Launching batch of {len(batch_tasks)} requests...")
                start_time = time.time()
                
                # Execute batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                batch_time = time.time() - start_time
                print(f"Batch completed in {batch_time:.2f}s")
                
                # Process results and handle exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        # Create error metrics for exceptions
                        error_metrics = RequestMetrics(
                            concurrent_requests=concurrent_requests,
                            text_length=0,
                            text_content="Error occurred",
                            start_time=start_time,
                            first_chunk_time=None,
                            end_time=time.time(),
                            total_time=batch_time,
                            time_to_first_chunk=None,
                            success=False,
                            error_message=str(result),
                            audio_size_bytes=0,
                            status_code=None,
                            chunks_received=0
                        )
                        tasks.append(error_metrics)
                    else:
                        tasks.append(result)
                
                # Brief pause between batches to avoid overwhelming the server
                if i + batch_size < requests_per_level:
                    await asyncio.sleep(1)
        
        return tasks
    
    def calculate_metrics(self, results: List[RequestMetrics]) -> ConcurrencyMetrics:
        """Calculate aggregated metrics for a concurrency level"""
        if not results:
            return None
            
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        total_times = [r.total_time for r in results]
        first_chunk_times = [r.time_to_first_chunk for r in successful_results if r.time_to_first_chunk is not None]
        audio_sizes = [r.audio_size_bytes for r in successful_results]
        text_lengths = [r.text_length for r in results if r.text_length > 0]
        
        # Calculate percentiles
        total_times_sorted = sorted(total_times)
        p95_index = int(0.95 * len(total_times_sorted))
        p95_time = total_times_sorted[p95_index] if total_times_sorted else 0
        
        return ConcurrencyMetrics(
            concurrent_requests=results[0].concurrent_requests,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            avg_total_time=statistics.mean(total_times) if total_times else 0,
            avg_time_to_first_chunk=statistics.mean(first_chunk_times) if first_chunk_times else 0,
            min_total_time=min(total_times) if total_times else 0,
            max_total_time=max(total_times) if total_times else 0,
            median_total_time=statistics.median(total_times) if total_times else 0,
            p95_total_time=p95_time,
            avg_audio_size=statistics.mean(audio_sizes) if audio_sizes else 0,
            requests_per_second=len(successful_results) / statistics.mean(total_times) if total_times and statistics.mean(total_times) > 0 else 0,
            error_rate=(len(failed_results) / len(results)) * 100 if results else 0,
            avg_text_length=statistics.mean(text_lengths) if text_lengths else 0
        )
    
    def print_results(self, metrics: ConcurrencyMetrics, detailed_results: List[RequestMetrics]):
        """Print benchmark results"""
        print(f"\n📊 Results for {metrics.concurrent_requests} concurrent requests:")
        print(f"   Total Requests: {metrics.total_requests}")
        print(f"   ✅ Successful: {metrics.successful_requests}")
        print(f"   ❌ Failed: {metrics.failed_requests}")
        print(f"   🚨 Error Rate: {metrics.error_rate:.1f}%")
        print(f"   📝 Avg Text Length: {metrics.avg_text_length:.0f} words")
        
        print(f"\n⏱️  Timing Metrics:")
        print(f"   Avg Total Time: {metrics.avg_total_time:.2f}s")
        print(f"   Avg Time to First Chunk: {metrics.avg_time_to_first_chunk:.2f}s")
        print(f"   Min/Max/Median: {metrics.min_total_time:.2f}s / {metrics.max_total_time:.2f}s / {metrics.median_total_time:.2f}s")
        print(f"   95th Percentile: {metrics.p95_total_time:.2f}s")
        
        print(f"\n🔊 Audio Metrics:")
        print(f"   Avg Audio Size: {metrics.avg_audio_size/1024:.1f} KB")
        print(f"   Requests/Second: {metrics.requests_per_second:.2f}")
        
        # Show errors if any
        failed_results = [r for r in detailed_results if not r.success]
        if failed_results:
            print(f"\n❌ Error Details:")
            error_counts = {}
            for result in failed_results:
                error_type = result.error_message or "Unknown error"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error, count in error_counts.items():
                print(f"   {error}: {count} occurrences")
    
    def save_results_to_csv(self, all_results: List[List[RequestMetrics]], filename: str = "benchmark_results.csv"):
        """Save detailed results to CSV file"""
        print(f"\n💾 Saving detailed results to {filename}...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'concurrent_requests', 'text_length', 'text_content', 'start_time',
                'total_time', 'time_to_first_chunk', 'success', 'error_message',
                'audio_size_bytes', 'status_code', 'chunks_received'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for results in all_results:
                for result in results:
                    row = asdict(result)
                    # Remove fields that aren't in fieldnames
                    row = {k: v for k, v in row.items() if k in fieldnames}
                    writer.writerow(row)
        
        print(f"✅ Results saved to {filename}")
    
    def save_summary_to_csv(self, summary_metrics: List[ConcurrencyMetrics], filename: str = "benchmark_summary.csv"):
        """Save summary metrics to CSV file"""
        print(f"💾 Saving summary metrics to {filename}...")
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = list(asdict(summary_metrics[0]).keys()) if summary_metrics else []
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for metrics in summary_metrics:
                writer.writerow(asdict(metrics))
        
        print(f"✅ Summary saved to {filename}")
    
    async def run_full_benchmark(self, max_concurrent: int = 32, requests_per_level: int = 10):
        """Run complete benchmark suite"""
        print("🚀 Starting ChatterBox TTS Benchmark")
        print(f"   Target URL: {self.base_url}")
        print(f"   Max Concurrent Requests: {max_concurrent}")
        print(f"   Requests per Level: {requests_per_level}")
        print(f"   Test Texts: {len(self.test_texts)} different lengths")
        
        concurrency_levels = [2**i for i in range(int(max_concurrent).bit_length()) if 2**i <= max_concurrent]
        if 1 not in concurrency_levels:
            concurrency_levels.insert(0, 1)
        
        print(f"   Concurrency Levels: {concurrency_levels}")
        
        all_results = []
        summary_metrics = []
        
        for concurrent_requests in concurrency_levels:
            try:
                results = await self.run_concurrency_test(concurrent_requests, requests_per_level)
                all_results.append(results)
                
                metrics = self.calculate_metrics(results)
                if metrics:
                    summary_metrics.append(metrics)
                    self.print_results(metrics, results)
                else:
                    print(f"❌ No results for {concurrent_requests} concurrent requests")
                    
            except Exception as e:
                print(f"❌ Error during {concurrent_requests} concurrent requests: {e}")
                continue
        
        # Print final summary
        print(f"\n{'='*60}")
        print("📋 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"{'Concurrent':<12} {'Success%':<9} {'Avg Time':<10} {'First Chunk':<12} {'RPS':<8} {'Errors'}")
        print("-" * 60)
        
        for metrics in summary_metrics:
            success_rate = ((metrics.total_requests - metrics.failed_requests) / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
            print(f"{metrics.concurrent_requests:<12} {success_rate:<8.1f}% {metrics.avg_total_time:<9.2f}s {metrics.avg_time_to_first_chunk:<11.2f}s {metrics.requests_per_second:<7.1f} {metrics.failed_requests}")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_results_to_csv(all_results, f"benchmark_detailed_{timestamp}.csv")
        self.save_summary_to_csv(summary_metrics, f"benchmark_summary_{timestamp}.csv")
        
        print(f"\n✅ Benchmark completed! Check the CSV files for detailed results.")

async def main():
    parser = argparse.ArgumentParser(description="Benchmark ChatterBox TTS API")
    parser.add_argument("--url", default="http://localhost:6401", help="Base URL of the ChatterBox API")
    parser.add_argument("--max-concurrent", type=int, default=32, help="Maximum concurrent requests to test")
    parser.add_argument("--requests-per-level", type=int, default=10, help="Number of requests per concurrency level")
    
    args = parser.parse_args()
    
    benchmark = ChatterBoxBenchmark(base_url=args.url)
    
    try:
        await benchmark.run_full_benchmark(
            max_concurrent=args.max_concurrent,
            requests_per_level=args.requests_per_level
        )
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Handle event loop for different Python versions
    try:
        asyncio.run(main())
    except RuntimeError:
        # For older Python versions or environments with existing event loops
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()
