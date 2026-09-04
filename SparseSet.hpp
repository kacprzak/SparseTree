#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sparse
{

/**
 * A sparse-set backed collection of unsigned integer keys stored contiguously in memory.
 *
 * Provides O(1) `contains`, `insert`, and `erase` with cache-friendly O(n)
 * iteration, at the expense of additional memory proportional to the largest
 * key ever inserted. Erasure is performed via swap-and-pop, so iteration order
 * is not stable — any mutation may reorder elements. Call `sort` to restore
 * key order.
 *
 * @tparam Key Unsigned integer key type. Memory is allocated for every value
 *             in [0, max_key], so prefer the smallest type that fits your key
 *             range (e.g. `std::uint16_t`).
 */
template< std::unsigned_integral Key >
class Set final
{
public:
	using key_type        = Key;
	using value_type      = Key;
	using size_type       = std::size_t;
	using reference       = value_type&;
	using const_reference = const value_type&;

private:
	// Marks unused keys.
	static constexpr auto s_invalid = std::numeric_limits< key_type >::max();

	void swap_by_key( const key_type first, const key_type second )
	{
		std::swap( m_dense[ m_sparse[ first ] ], m_dense[ m_sparse[ second ] ] );
		std::swap( m_sparse[ first ], m_sparse[ second ] );
	}

	void swap( const size_type first, const size_type second )
	{
		std::swap( m_sparse[ m_dense[ first ] ], m_sparse[ m_dense[ second ] ] );
		std::swap( m_dense[ first ], m_dense[ second ] );
	}

	void do_insert( const key_type& key )
	{
		if( key >= m_sparse.size() )
			m_sparse.resize( key + 1u, s_invalid );

		m_sparse[ key ] = m_dense.size();
		m_dense.emplace_back( key );
	}

public:
	Set() = default;

	template< std::input_iterator IterT >
	Set( IterT first, IterT last )
	{
		reserve( std::distance( first, last ) );

		for( ; first != last; ++first )
			insert( *first );
	}

	Set( std::initializer_list< key_type > init ) : Set{ init.begin(), init.end() } {}

	[[nodiscard]] constexpr bool empty() const noexcept { return m_dense.empty(); }
	[[nodiscard]] constexpr size_type size() const noexcept { return m_dense.size(); }

	[[nodiscard]]
	constexpr bool contains( const key_type& key ) const noexcept
	{
		return key < m_sparse.size() and m_sparse[ key ] != s_invalid;
	}

	constexpr void clear() noexcept
	{
		m_dense.clear();
		m_sparse.clear();
	}

	constexpr void reserve( size_type n )
	{
		m_dense.reserve( n );
		m_sparse.reserve( n );
	}

	/**
	 * Returns the smallest key that is not currently in use.
	 *
	 * Useful for allocating the next available key without manual tracking.
	 * If all keys in [0, size-1] are occupied, returns `size`, which is a
	 * valid key for the next insertion.
	 */
	[[nodiscard]]
	constexpr key_type find_slot() const
	{
		const auto it = std::find( m_sparse.begin(), m_sparse.end(), s_invalid );

		return std::distance( m_sparse.begin(), it );
	}

	/**
	 * Inserts @p key into the set if it is not already present.
	 *
	 * @param  key Key to insert. Must not equal the sentinel value
	 *             (`std::numeric_limits<key_type>::max()`).
	 * @returns `true` if the key was inserted; `false` if it already existed.
	 * @throws std::logic_error if @p key is the sentinel value.
	 */
	bool insert( const key_type& key )
	{
		if( key == s_invalid )
			throw std::logic_error{ "Invalid key value." };

		if( contains( key ) )
			return false;

		do_insert( key );
		return true;
	}

	/**
	 * Erases @p key from the set, if present.
	 *
	 * Uses swap-and-pop internally, so the erased slot is filled by the last
	 * element. Iteration order is not preserved after erasure.
	 *
	 * @param  key Key to erase.
	 * @returns `true` if the key was erased; `false` if it was not found.
	 */
	bool erase( const key_type& key )
	{
		if( not contains( key ) )
			return false;

		// Swap and pop
		swap_by_key( key, m_dense.back() );

		m_dense.pop_back();
		m_sparse[ key ] = s_invalid;

		return true;
	}

	/**
	 * Sorts the dense storage in ascending key order.
	 *
	 * After this call, iterating via `begin()`/`end()` visits keys in ascending
	 * order. Invalidates all iterators. Complexity: O(n log n).
	 */
	constexpr void sort()
	{
		std::sort( m_dense.begin(), m_dense.end() );

		for( std::size_t i = 0; i < m_dense.size(); ++i )
		{
			std::size_t curr = i;
			std::size_t next = m_sparse[ m_dense[ i ] ];

			while( curr != next )
			{
				m_sparse[ m_dense[ curr ] ] = curr;
				curr                        = next;
				next                        = m_sparse[ m_dense[ curr ] ];
			}
		}
	}

	/**
	 * Reorders dense storage according to the given index permutation.
	 *
	 * @p permutation must be the same size as the current number of elements,
	 * where `permutation[i]` is the index that element `i` should move to.
	 * The permutation is applied in-place and consumed (modified) during the call.
	 *
	 * @param permutation A span of destination indices, one per element.
	 * @pre   `permutation.size() == size()`.
	 */
	constexpr void reorder( std::span< size_type > permutation )
	{
		assert( permutation.size() == m_dense.size() );

		for( std::size_t i = 0; i < permutation.size(); ++i )
		{
			std::size_t curr = i;
			std::size_t next = permutation[ i ];

			while( curr != next )
			{
				swap( permutation[ curr ], permutation[ next ] );
				permutation[ curr ] = curr;
				curr                = next;
				next                = permutation[ curr ];
			}
		}
	}

	[[nodiscard]] constexpr auto begin() const noexcept { return m_dense.begin(); }
	[[nodiscard]] constexpr auto end() const noexcept { return m_dense.end(); }

	/**
	 * Returns `true` if both sets contain exactly the same keys, regardless of insertion order.
	 */
	[[nodiscard]]
	friend bool operator==( const Set& lhs, const Set& rhs ) noexcept
	{
		if( lhs.size() != rhs.size() )
			return false;

		for( const auto key : lhs )
			if( not rhs.contains( key ) )
				return false;

		return true;
	}

private:
	std::vector< key_type > m_sparse; // maps key -> dense index
	std::vector< key_type > m_dense;  // dense array of keys
};

} // namespace sparse
