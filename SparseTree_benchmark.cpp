#include <benchmark/benchmark.h>

#include <SparseTree.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace
{

auto createRandomSet( std::size_t size )
{
	sparse::Tree< std::uint16_t, float > tree;
	tree.reserve( size );

	tree.insert( 0u, float{} );

	for( auto i = 1u; i < size; ++i )
		tree.insert( i, i, std::rand() % tree.size() );

	return tree;
}

class BM_SparseTree : public ::benchmark::Fixture
{
public:
	void SetUp( const ::benchmark::State& st ) override { tree = createRandomSet( st.range( 0 ) ); }
	void TearDown( const ::benchmark::State& ) override { tree.clear(); }

	sparse::Tree< std::uint16_t, float > tree;
};

class BM_SparseTreeSorted : public ::benchmark::Fixture
{
public:
	void SetUp( const ::benchmark::State& st ) override
	{
		tree = createRandomSet( st.range( 0 ) );
		tree.sort_bfs();
	}
	void TearDown( const ::benchmark::State& ) override { tree.clear(); }

	sparse::Tree< std::uint16_t, float > tree;
};

} // namespace

BENCHMARK_DEFINE_F( BM_SparseTree, erase )( benchmark::State& state )
{
	const auto size = state.range( 0 );

	for( auto _ : state )
	{
		for( auto i = 0u; i < size; ++i )
		{
			auto count = tree.erase( std::rand() % size );
			benchmark::DoNotOptimize( count );
		}
	}

	state.SetItemsProcessed( state.iterations() * size );
	state.SetComplexityN( size );
}
BENCHMARK_REGISTER_F( BM_SparseTree, erase )->Range( 8, 8 << 10 )->Complexity();

BENCHMARK_DEFINE_F( BM_SparseTree, for_each_bfs )( benchmark::State& state )
{
	const auto size = state.range( 0 );

	for( auto _ : state )
	{
		float sum = 0.f;
		tree.for_each_bfs( [ &sum ]( const auto& kv ) { sum += kv.second; } );
		benchmark::DoNotOptimize( sum );
	}

	state.SetItemsProcessed( state.iterations() * size );
	state.SetComplexityN( size );
}
BENCHMARK_REGISTER_F( BM_SparseTree, for_each_bfs )->Range( 8, 8 << 10 )->Complexity();

BENCHMARK_DEFINE_F( BM_SparseTree, for_each_dfs )( benchmark::State& state )
{
	const auto size = state.range( 0 );

	for( auto _ : state )
	{
		float sum = 0.f;
		tree.for_each_dfs(
		    [ &sum ]( const auto& kv )
		    {
			    sum += kv.second;
			    return true;
		    },
		    []( const auto& /*kv*/ ) {} );
		benchmark::DoNotOptimize( sum );
	}

	state.SetItemsProcessed( state.iterations() * size );
	state.SetComplexityN( size );
}
BENCHMARK_REGISTER_F( BM_SparseTree, for_each_dfs )->Range( 8, 8 << 10 )->Complexity();

BENCHMARK_DEFINE_F( BM_SparseTreeSorted, for_each )( benchmark::State& state )
{
	const auto size = state.range( 0 );

	for( auto _ : state )
	{
		float sum = 0.f;
		for( const auto& kv : tree )
			sum += kv.second;
		benchmark::DoNotOptimize( sum );
	}

	state.SetItemsProcessed( state.iterations() * size );
	state.SetComplexityN( size );
}
BENCHMARK_REGISTER_F( BM_SparseTreeSorted, for_each )->Range( 8, 8 << 10 )->Complexity();

BENCHMARK_DEFINE_F( BM_SparseTree, sort_bfs )( benchmark::State& state )
{
	const auto size = state.range( 0 );

	for( auto _ : state )
	{
		float sum = 0.f;
		tree.sort_bfs();

		for( const auto& kv : tree )
			sum += kv.second;
		benchmark::DoNotOptimize( sum );
	}

	state.SetItemsProcessed( state.iterations() * size );
	state.SetComplexityN( size );
}
BENCHMARK_REGISTER_F( BM_SparseTree, sort_bfs )->Range( 8, 8 << 10 )->Complexity();
