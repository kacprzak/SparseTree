#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <ranges>

#include "SparseTree.hpp"

namespace
{

TEST( SparseTree, ctor )
{
	sparse::Tree< std::uint8_t, float > tree;

	EXPECT_TRUE( tree.empty() );
}

TEST( SparseTree, initializer_list )
{
	sparse::Tree< std::uint16_t, float > tree{ { 0, 0.f }, { 1, 1.f }, { 2, 2.f } };

	EXPECT_FALSE( tree.empty() );
	EXPECT_EQ( tree.size(), 3u );

	int counter = 0;
	tree.for_each_bfs( [ & ]( const auto& /*kv*/ ) { ++counter; } );
	EXPECT_EQ( counter, 3 );
}

TEST( SparseTree, iterator )
{
	using namespace std::ranges;

	sparse::Tree< std::uint16_t, float > tree{ { 0, 0.f }, { 1, 1.f }, { 2, 2.f } };

	for( const auto& [ k, v ] : tree )
		v = -v;

	const std::vector< float > expected{ -0.f, -1.f, -2.f };

	EXPECT_TRUE( equal( views::values( tree ), expected ) );
}

TEST( SparseTree, insert )
{
	sparse::Tree< std::uint8_t, float > tree;

	EXPECT_TRUE( tree.insert( 0, 2.f ) );
	EXPECT_FALSE( tree.empty() );
	EXPECT_EQ( tree.at( 0 ), 2.f );
}

TEST( SparseTree, erase )
{
	sparse::Tree< std::uint8_t, float > tree;

	EXPECT_TRUE( tree.insert( 0, 2.f ) );
	EXPECT_TRUE( tree.contains( 0 ) );
	EXPECT_EQ( tree.erase( 0 ), 1u );
	EXPECT_FALSE( tree.contains( 0 ) );
}

TEST( SparseTree, parent )
{
	sparse::Tree< std::uint8_t, float > tree;

	EXPECT_TRUE( tree.insert( 0, 0.f ) );
	EXPECT_EQ( tree.parent( 0 ), tree.end() );

	EXPECT_TRUE( tree.insert( 1, 1.f, 0 ) );
	EXPECT_EQ( ( *tree.parent( 1 ) ).first, 0u );
	EXPECT_EQ( ( *tree.parent( 1 ) ).second, 0.f );
}

TEST( SparseTree, children )
{
	sparse::Tree< std::uint8_t, float > tree;

	// 0
	// ├── 3
	// ├── 2
	// │   ├── 5
	// │   └── 4
	// └── 1
	EXPECT_TRUE( tree.insert( 0, 0.f ) );
	EXPECT_TRUE( tree.insert( 3, 3.f, 0 ) );
	EXPECT_TRUE( tree.insert_after( 2, 2.f, 3 ) );
	EXPECT_TRUE( tree.insert_after( 1, 1.f, 2 ) );
	EXPECT_TRUE( tree.insert( 4, 4.f, 2 ) );
	EXPECT_TRUE( tree.insert( 5, 5.f, 2 ) );
	EXPECT_FALSE( tree.insert( 5, -5.f, 2 ) ); // duplicate key

	{
		const std::vector< float > expected{ 3.f, 2.f, 1.f };

		std::vector< float > result;
		for( auto it = tree.children_begin( 0 ); it != tree.end(); it = tree.children_next( it ) )
			result.push_back( it->second );

		EXPECT_EQ( result, expected );
	}

	EXPECT_EQ( tree.size(), 6u );
	EXPECT_EQ( tree.erase( 2 ), 3u ); // erases node 2 and its subtree (4, 5)
	EXPECT_EQ( tree.size(), 3u );

	{
		const std::vector< float > expected{ 3.f, 1.f };

		std::vector< float > result;
		for( auto it = tree.children_begin( 0 ); it != tree.end(); it = tree.children_next( it ) )
			result.push_back( it->second );

		EXPECT_EQ( result, expected );
	}
}

namespace
{
void init( sparse::Tree< std::uint8_t, float >& tree )
{
	// Forest (root-list order: 7, then 0):
	// 7  (leaf)
	// 0
	// ├── 1
	// │   ├── 4
	// │   └── 5
	// ├── 2
	// │   └── 6
	// └── 3
	EXPECT_TRUE( tree.insert( 0, 0.f ) );
	EXPECT_TRUE( tree.insert( 1, 1.f, 0 ) );
	EXPECT_TRUE( tree.insert_after( 2, 2.f, 1 ) );
	EXPECT_TRUE( tree.insert_after( 3, 3.f, 2 ) );
	EXPECT_TRUE( tree.insert( 6, 6.f, 2 ) );
	EXPECT_TRUE( tree.insert( 4, 4.f, 1 ) );
	EXPECT_TRUE( tree.insert_after( 5, 5.f, 4 ) );
	EXPECT_TRUE( tree.insert( 7, 7.f ) ); // prepended → becomes first root
}
} // namespace

TEST( SparseTree, for_each_bfs )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Roots first (7, 0), then each level left-to-right.
	const std::vector< float > expected{ 7.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

	std::vector< float > result;
	tree.for_each_bfs( [ & ]( const auto& kv ) { result.push_back( kv.second ); } );

	EXPECT_EQ( result, expected );
}

TEST( SparseTree, for_each_dfs )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Pre-order: each node visited before its children, roots in root-list order.
	const std::vector< float > expected{ 7.f, 0.f, 1.f, 4.f, 5.f, 2.f, 6.f, 3.f };

	std::vector< float > result;
	tree.for_each_dfs(
	    [ & ]( const auto& kv )
	    {
		    result.push_back( kv.second );
		    return true;
	    },
	    []( const auto& ) {} );

	EXPECT_EQ( result, expected );
}

TEST( SparseTree, for_each_bfs_from )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Subtree rooted at node 0
	{
		const std::vector< float > expected{ 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

		std::vector< float > result;
		tree.for_each_bfs( 0, [ & ]( const auto& kv ) { result.push_back( kv.second ); } );

		EXPECT_EQ( result, expected );
	}

	// Subtree rooted at node 1
	{
		const std::vector< float > expected{ 1.f, 4.f, 5.f };

		std::vector< float > result;
		tree.for_each_bfs( 1, [ & ]( const auto& kv ) { result.push_back( kv.second ); } );

		EXPECT_EQ( result, expected );
	}

	// Node 7 (leaf)
	{
		const std::vector< float > expected{ 7.f };

		std::vector< float > result;
		tree.for_each_bfs( 7, [ & ]( const auto& kv ) { result.push_back( kv.second ); } );

		EXPECT_EQ( result, expected );
	}
}

TEST( SparseTree, for_each_dfs_from )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Subtree rooted at node 0
	{
		const std::vector< float > expected{ 0.f, 1.f, 4.f, 5.f, 2.f, 6.f, 3.f };

		std::vector< float > result;
		tree.for_each_dfs(
		    0,
		    [ & ]( const auto& kv )
		    {
			    result.push_back( kv.second );
			    return true;
		    },
		    []( const auto& ) {} );

		EXPECT_EQ( result, expected );
	}

	// Subtree rooted at node 2
	{
		const std::vector< float > expected{ 2.f, 6.f };

		std::vector< float > result;
		tree.for_each_dfs(
		    2,
		    [ & ]( const auto& kv )
		    {
			    result.push_back( kv.second );
			    return true;
		    },
		    []( const auto& ) {} );

		EXPECT_EQ( result, expected );
	}

	// on_enter returning false skips the subtree; on_leave still fires for that node.
	{
		std::vector< float > entered;
		std::vector< float > left;

		tree.for_each_dfs(
		    0,
		    [ & ]( const auto& kv )
		    {
			    entered.push_back( kv.second );
			    return kv.second == 0.f; // descend only into node 0
		    },
		    [ & ]( const auto& kv ) { left.push_back( kv.second ); } );

		const std::vector< float > expected_entered{ 0.f, 1.f, 2.f, 3.f };
		const std::vector< float > expected_left{ 1.f, 2.f, 3.f, 0.f }; // children leave before parent
		EXPECT_EQ( entered, expected_entered );
		EXPECT_EQ( left, expected_left );
	}
}

TEST( SparseTree, set_parent_to_node )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Before: 0 → {1, 2, 3}; 7 is a root
	// Move root 7 under node 2 → 7 should become a child of 2
	EXPECT_TRUE( tree.set_parent( 7, 2 ) );
	EXPECT_EQ( ( *tree.parent( 7 ) ).first, 2u );

	// Children of 2 are now: 7 (prepended), then 6
	std::vector< uint8_t > children_of_2;
	for( auto it = tree.children_begin( 2 ); it != tree.end(); it = tree.children_next( it ) )
		children_of_2.push_back( it->first );
	EXPECT_EQ( children_of_2, ( std::vector< uint8_t >{ 7, 6 } ) );

	// 7 is no longer a root
	std::vector< float > roots;
	tree.for_each_bfs(
	    [ & ]( const auto& kv )
	    {
		    if( tree.parent( kv.first ) == tree.end() )
			    roots.push_back( kv.second );
	    } );
	for( float v : roots )
		EXPECT_NE( v, 7.f );
}

TEST( SparseTree, set_parent_to_root )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Move node 1 (child of 0) to become a root
	EXPECT_TRUE( tree.unset_parent( 1 ) );
	EXPECT_EQ( tree.parent( 1 ), tree.end() );

	// Node 0's children should no longer include 1
	std::vector< uint8_t > children_of_0;
	for( auto it = tree.children_begin( 0 ); it != tree.end(); it = tree.children_next( it ) )
		children_of_0.push_back( it->first );
	for( auto k : children_of_0 )
		EXPECT_NE( k, 1 );

	// Subtree of 1 (nodes 4, 5) moves with it
	std::vector< float > subtree;
	tree.for_each_bfs( 1, [ & ]( const auto& kv ) { subtree.push_back( kv.second ); } );
	EXPECT_EQ( subtree, ( std::vector< float >{ 1.f, 4.f, 5.f } ) );
}

TEST( SparseTree, set_parent_already_root_noop )
{
	sparse::Tree< std::uint8_t, float > tree;
	EXPECT_TRUE( tree.insert( 0, 0.f ) );
	EXPECT_TRUE( tree.unset_parent( 0 ) ); // already a root — no-op, returns true
	EXPECT_EQ( tree.parent( 0 ), tree.end() );
}

TEST( SparseTree, set_parent_cycle_rejected )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Trying to make node 0 a child of node 4 (which is a descendant of 0) must fail
	EXPECT_FALSE( tree.set_parent( 0, 4 ) );
	// Structure unchanged: 4 is still a child of 1 which is a child of 0
	EXPECT_EQ( ( *tree.parent( 4 ) ).first, 1u );
	EXPECT_EQ( ( *tree.parent( 0 ) ), ( *tree.end() ) );
}

TEST( SparseTree, set_parent_self_rejected )
{
	sparse::Tree< std::uint8_t, float > tree;
	EXPECT_TRUE( tree.insert( 0, 0.f ) );
	EXPECT_FALSE( tree.set_parent( 0, 0 ) );
}

TEST( SparseTree, set_parent_missing_key_rejected )
{
	sparse::Tree< std::uint8_t, float > tree;
	EXPECT_TRUE( tree.insert( 0, 0.f ) );
	EXPECT_FALSE( tree.set_parent( 99, 0 ) );
	EXPECT_FALSE( tree.set_parent( 0, 99 ) );
	EXPECT_FALSE( tree.unset_parent( 99 ) );
}

TEST( SparseTree, reverse_children )
{
	sparse::Tree< std::uint8_t, float > tree;

	// 0
	// ├── 1
	// ├── 2
	// └── 3
	EXPECT_TRUE( tree.insert( 0, 0.f ) );
	EXPECT_TRUE( tree.insert( 1, 1.f, 0 ) );
	EXPECT_TRUE( tree.insert_after( 2, 2.f, 1 ) );
	EXPECT_TRUE( tree.insert_after( 3, 3.f, 2 ) );

	{
		const std::vector< float > expected{ 1.f, 2.f, 3.f };
		std::vector< float > result;
		for( auto it = tree.children_begin( 0 ); it != tree.end(); it = tree.children_next( it ) )
			result.push_back( it->second );
		EXPECT_EQ( result, expected );
	}

	tree.reverse_children( 0 );

	{
		const std::vector< float > expected{ 3.f, 2.f, 1.f };
		std::vector< float > result;
		for( auto it = tree.children_begin( 0 ); it != tree.end(); it = tree.children_next( it ) )
			result.push_back( it->second );
		EXPECT_EQ( result, expected );
	}

	// Reversing a single child is a no-op.
	EXPECT_TRUE( tree.insert( 4, 4.f, 1 ) );
	tree.reverse_children( 1 );
	EXPECT_EQ( tree.children_begin( 1 )->first, 4u );

	// Reversing an empty child list is a no-op.
	tree.reverse_children( 3 );
	EXPECT_EQ( tree.children_begin( 3 ), tree.end() );
}

TEST( SparseTree, sort_bfs )
{
	using namespace std::ranges;

	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	{
		const std::vector< float > expected{ 7.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };

		tree.sort_bfs();

		EXPECT_TRUE( equal( views::values( tree ), expected ) );
	}
}

TEST( SparseTree, for_each_bfs_reentrant )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Outer BFS collects the keys of nodes at level 1 (children of node 0).
	// For each such node, an inner BFS collects its subtree values.
	// This verifies that nested for_each_bfs calls don't corrupt each other.
	std::vector< std::vector< float > > subtrees;

	tree.for_each_bfs( 0,
	                   [ & ]( const auto& kv )
	                   {
		                   std::vector< float > sub;
		                   tree.for_each_bfs( kv.first, [ & ]( const auto& inner ) { sub.push_back( inner.second ); } );
		                   subtrees.push_back( std::move( sub ) );
	                   } );

	// Subtree of 0: {0,1,2,3,4,5,6}  Subtree of 1: {1,4,5}  Subtree of 2: {2,6}  Subtree of 3: {3}
	ASSERT_EQ( subtrees.size(), 7u );
	EXPECT_EQ( subtrees[ 0 ], ( std::vector< float >{ 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f } ) );
	EXPECT_EQ( subtrees[ 1 ], ( std::vector< float >{ 1.f, 4.f, 5.f } ) );
	EXPECT_EQ( subtrees[ 2 ], ( std::vector< float >{ 2.f, 6.f } ) );
	EXPECT_EQ( subtrees[ 3 ], ( std::vector< float >{ 3.f } ) );
	EXPECT_EQ( subtrees[ 4 ], ( std::vector< float >{ 4.f } ) );
	EXPECT_EQ( subtrees[ 5 ], ( std::vector< float >{ 5.f } ) );
	EXPECT_EQ( subtrees[ 6 ], ( std::vector< float >{ 6.f } ) );
}

TEST( SparseTree, for_each_dfs_reentrant )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// For each node visited in an outer DFS, run an inner BFS over the same node's subtree.
	// This verifies that DFS and BFS scratch are independent and can be nested freely.
	std::vector< float > outer_enter_order;
	std::vector< float > inner_root_values;

	tree.for_each_dfs(
	    0,
	    [ & ]( const auto& kv )
	    {
		    outer_enter_order.push_back( kv.second );

		    float first = -1.f;
		    tree.for_each_bfs( kv.first,
		                       [ & ]( const auto& inner )
		                       {
			                       if( first < 0.f )
				                       first = inner.second;
		                       } );
		    inner_root_values.push_back( first );

		    return true;
	    },
	    []( const auto& ) {} );

	// on_enter visits in DFS pre-order; the inner BFS of each subtree starts at that same node.
	const std::vector< float > expected{ 0.f, 1.f, 4.f, 5.f, 2.f, 6.f, 3.f };
	EXPECT_EQ( outer_enter_order, expected );
	EXPECT_EQ( inner_root_values, expected ); // each inner BFS starts at the same node
}

TEST( SparseTree, sub_tree )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Subtree rooted at node 0 (excludes the root 7)
	{
		auto sub = tree.sub_tree( 0 );

		EXPECT_EQ( sub.size(), 7u );
		EXPECT_EQ( sub.parent( 0 ), sub.end() ); // 0 is a root in the new tree

		const std::vector< float > expected{ 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
		std::vector< float > result;
		sub.for_each_bfs( [ & ]( const auto& kv ) { result.push_back( kv.second ); } );
		EXPECT_EQ( result, expected );
	}

	// Subtree rooted at node 1
	{
		auto sub = tree.sub_tree( 1 );

		EXPECT_EQ( sub.size(), 3u );
		EXPECT_EQ( sub.parent( 1 ), sub.end() );
		EXPECT_EQ( ( *sub.parent( 4 ) ).first, 1u );
		EXPECT_EQ( ( *sub.parent( 5 ) ).first, 1u );

		const std::vector< float > expected{ 1.f, 4.f, 5.f };
		std::vector< float > result;
		sub.for_each_bfs( [ & ]( const auto& kv ) { result.push_back( kv.second ); } );
		EXPECT_EQ( result, expected );
	}

	// Leaf node
	{
		auto sub = tree.sub_tree( 7 );

		EXPECT_EQ( sub.size(), 1u );
		EXPECT_EQ( sub.parent( 7 ), sub.end() );
	}

	// Original tree is unmodified
	EXPECT_EQ( tree.size(), 8u );
}

TEST( SparseTree, sub_trees_disjoint )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Subtrees rooted at 1 and 7 are disjoint
	auto result = tree.sub_trees( { 1, 7 } );
	ASSERT_TRUE( result.has_value() );

	EXPECT_EQ( result->size(), 4u ); // 1, 4, 5, 7

	// Both requested roots are roots in the result
	EXPECT_EQ( result->parent( 1 ), result->end() );
	EXPECT_EQ( result->parent( 7 ), result->end() );

	// Sub-structure preserved
	EXPECT_EQ( ( *result->parent( 4 ) ).first, 1u );
	EXPECT_EQ( ( *result->parent( 5 ) ).first, 1u );

	// BFS order: roots in root-list order (last inserted = first root), then their children
	const std::vector< float > expected{ 7.f, 1.f, 4.f, 5.f };
	std::vector< float > bfs;
	result->for_each_bfs( [ & ]( const auto& kv ) { bfs.push_back( kv.second ); } );
	EXPECT_EQ( bfs, expected );

	// Original tree unchanged
	EXPECT_EQ( tree.size(), 8u );
}

TEST( SparseTree, sub_trees_overlap_fails )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	// Node 1 is a descendant of node 0 → overlap
	EXPECT_FALSE( tree.sub_trees( { 0, 1 } ).has_value() );
	EXPECT_FALSE( tree.sub_trees( { 1, 0 } ).has_value() );

	// Duplicate key
	EXPECT_FALSE( tree.sub_trees( { 1, 1 } ).has_value() );
}

TEST( SparseTree, sub_trees_missing_key_fails )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	EXPECT_FALSE( tree.sub_trees( { 1, 99 } ).has_value() );
}

TEST( SparseTree, sub_trees_range )
{
	sparse::Tree< std::uint8_t, float > tree;
	init( tree );

	const std::vector< std::uint8_t > roots{ 3, 7 };
	auto result = tree.sub_trees( roots );
	ASSERT_TRUE( result.has_value() );

	EXPECT_EQ( result->size(), 2u );
	EXPECT_EQ( result->parent( 3 ), result->end() );
	EXPECT_EQ( result->parent( 7 ), result->end() );
}

TEST( SparseTree, insert_move_only )
{
	sparse::Tree< std::uint8_t, std::unique_ptr< int > > tree;

	EXPECT_TRUE( tree.insert( 0u, std::make_unique< int >( 1 ) ) );
	EXPECT_TRUE( tree.insert( 1u, std::make_unique< int >( 2 ), 0u ) );
	EXPECT_TRUE( tree.insert_after( 2u, std::make_unique< int >( 3 ), 1u ) );

	EXPECT_EQ( tree.size(), 3u );
	EXPECT_EQ( *tree.at( 0u ), 1 );
	EXPECT_EQ( *tree.at( 1u ), 2 );
	EXPECT_EQ( *tree.at( 2u ), 3 );
}

TEST( SparseTree, equality_same_structure )
{
	sparse::Tree< std::uint8_t, float > a;
	a.insert( 0, 0.f );
	a.insert( 1, 1.f, 0 );
	a.insert( 2, 2.f, 0 );

	sparse::Tree< std::uint8_t, float > b;
	b.insert( 0, 0.f );
	b.insert( 1, 1.f, 0 );
	b.insert( 2, 2.f, 0 );

	EXPECT_EQ( a, b );
}

TEST( SparseTree, equality_different_values )
{
	sparse::Tree< std::uint8_t, float > a;
	a.insert( 0, 0.f );
	a.insert( 1, 1.f, 0 );

	sparse::Tree< std::uint8_t, float > b;
	b.insert( 0, 0.f );
	b.insert( 1, 99.f, 0 );

	EXPECT_NE( a, b );
}

TEST( SparseTree, equality_different_topology )
{
	// Same nodes and values, but different parent-child relationships.
	sparse::Tree< std::uint8_t, float > a;
	a.insert( 0, 0.f );
	a.insert( 1, 1.f, 0 ); // 1 is child of 0

	sparse::Tree< std::uint8_t, float > b;
	b.insert( 0, 0.f );
	b.insert( 1, 1.f ); // 1 is a root (no parent)

	EXPECT_NE( a, b );
}

TEST( SparseTree, equality_empty )
{
	sparse::Tree< std::uint8_t, float > a;
	sparse::Tree< std::uint8_t, float > b;

	EXPECT_EQ( a, b );
}

} // namespace
