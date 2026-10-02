// SPDX-License-Identifier: GPL-2.0
/*
 * fs/ext4/verity.c: fs-verity support for ext4
 *
 * Copyright 2019 Google LLC
 */

/*
 * Implementation of fsverity_operations for ext4.
 *
 * ext4 stores the verity metadata (Merkle tree and fsverity_descriptor) past
 * the end of the file, starting at the first 64K boundary beyond i_size.  This
 * approach works because (a) verity files are readonly, and (b) pages fully
 * beyond i_size aren't visible to userspace but can be read/written internally
 * by ext4 with only some relatively small changes to ext4.  This approach
 * avoids having to depend on the EA_INODE feature and on rearchitecturing
 * ext4's xattr support to support paging multi-gigabyte xattrs into memory, and
 * to support encrypting xattrs.  Note that the verity metadata *must* be
 * encrypted when the file is, since it contains hashes of the plaintext data.
 *
 * Using a 64K boundary rather than a 4K one keeps things ready for
 * architectures with 64K pages, and it doesn't necessarily waste space on-disk
 * since there can be a hole between i_size and the start of the Merkle tree.
 */

#include <linux/quotaops.h>
#include <linux/bio.h>
#include <linux/completion.h>
#include <linux/pagemap.h>
#include <linux/highmem.h>

#include "ext4.h"
#include "ext4_extents.h"
#include "ext4_jbd2.h"

static inline loff_t ext4_verity_metadata_pos(const struct inode *inode)
{
	return round_up(inode->i_size, 65536);
}

/*
 * Read some verity metadata from the inode.  __vfs_read() can't be used because
 * we need to read beyond i_size.
 */
static int pagecache_read(struct inode *inode, void *buf, size_t count,
			  loff_t pos)
{
	while (count) {
		size_t n = min_t(size_t, count,
				 PAGE_SIZE - offset_in_page(pos));
		struct page *page;
		void *addr;

		page = read_mapping_page(inode->i_mapping, pos >> PAGE_SHIFT,
					 NULL);
		if (IS_ERR(page))
			return PTR_ERR(page);

		addr = kmap_atomic(page);
		memcpy(buf, addr + offset_in_page(pos), n);
		kunmap_atomic(addr);

		put_page(page);

		buf += n;
		pos += n;
		count -= n;
	}
	return 0;
}

/*
 * Write some verity metadata to the inode for FS_IOC_ENABLE_VERITY.
 * kernel_write() can't be used because the file descriptor is readonly.
 */
/*
 * 3.4 has no post-EOF pagecache writeback.  Journal verity metadata through
 * buffer heads instead, using ext4's existing directory-block allocation
 * path.  This preserves i_size and i_disksize, and orphan recovery removes
 * the blocks after an interrupted enable operation.  Reads of these pages
 * use the same buffer cache, including metadata not yet checkpointed.
 */
static int pagecache_write(struct inode *inode, const void *buf, size_t count,
			  loff_t pos)
{
	if (pos < 0 || pos > inode->i_sb->s_maxbytes - count)
		return -EFBIG;
	while (count) {
		size_t n = min_t(size_t, count, PAGE_SIZE - offset_in_page(pos));
		handle_t *handle;
		struct buffer_head *bh;
		int err, stop_err;

		handle = ext4_journal_start(inode,
				ext4_chunk_trans_blocks(inode, 1) + 1);
		if (IS_ERR(handle))
			return PTR_ERR(handle);
		bh = ext4_bread(handle, inode, pos >> PAGE_SHIFT, 1, &err);
		if (!bh) {
			err = err ?: -EIO;
			goto stop;
		}
		err = ext4_journal_get_write_access(handle, bh);
		if (!err) {
			lock_buffer(bh);
			memcpy(bh->b_data + offset_in_page(pos), buf, n);
			set_buffer_uptodate(bh);
			unlock_buffer(bh);
			err = ext4_handle_dirty_metadata(handle, inode, bh);
			if (!err && !ext4_handle_valid(handle))
				err = sync_dirty_buffer(bh);
		}
		brelse(bh);
stop:
		stop_err = ext4_journal_stop(handle);
		if (err || stop_err)
			return err ?: stop_err;
		buf += n;
		pos += n;
		count -= n;
	}
	return 0;
}

static int ext4_begin_enable_verity(struct file *filp)
{
	struct inode *inode = file_inode(filp);
	const int credits = 2; /* superblock and inode for ext4_orphan_add() */
	handle_t *handle;
	int err;

	if (ext4_verity_in_progress(inode))
		return -EBUSY;

	if (!EXT4_HAS_RO_COMPAT_FEATURE(inode->i_sb,
					EXT4_FEATURE_RO_COMPAT_VERITY) ||
	    inode->i_sb->s_blocksize != PAGE_SIZE)
		return -EOPNOTSUPP;

	/* Stabilize data before hashing; metadata itself is journalled. */
	err = filemap_write_and_wait(inode->i_mapping);
	if (err)
		return err;

	dquot_initialize(inode);

	if (!ext4_test_inode_flag(inode, EXT4_INODE_EXTENTS)) {
		ext4_warning(inode->i_sb, "verity requires extent-based files");
		return -EOPNOTSUPP;
	}

	/*
	 * ext4 uses the last allocated block to find the verity descriptor, so
	 * we must remove any other blocks past EOF which might confuse things.
	 */
	ext4_truncate(inode);

	handle = ext4_journal_start(inode, credits);
	if (IS_ERR(handle))
		return PTR_ERR(handle);

	err = ext4_orphan_add(handle, inode);
	if (err == 0)
		ext4_set_inode_state(inode, EXT4_STATE_VERITY_IN_PROGRESS);

	ext4_journal_stop(handle);
	return err;
}

/*
 * ext4 stores the verity descriptor beginning on the next filesystem block
 * boundary after the Merkle tree.  Then, the descriptor size is stored in the
 * last 4 bytes of the last allocated filesystem block --- which is either the
 * block in which the descriptor ends, or the next block after that if there
 * weren't at least 4 bytes remaining.
 *
 * We can't simply store the descriptor in an xattr because it *must* be
 * encrypted when ext4 encryption is used, but ext4 encryption doesn't encrypt
 * xattrs.  Also, if the descriptor includes a large signature blob it may be
 * too large to store in an xattr without the EA_INODE feature.
 */
static int ext4_write_verity_descriptor(struct inode *inode, const void *desc,
					size_t desc_size, u64 merkle_tree_size)
{
	const u64 desc_pos = round_up(ext4_verity_metadata_pos(inode) +
				      merkle_tree_size, inode->i_sb->s_blocksize);
	const u64 desc_end = desc_pos + desc_size;
	const __le32 desc_size_disk = cpu_to_le32(desc_size);
	const u64 desc_size_pos = round_up(desc_end + sizeof(desc_size_disk),
					   inode->i_sb->s_blocksize) -
				  sizeof(desc_size_disk);
	int err;

	err = pagecache_write(inode, desc, desc_size, desc_pos);
	if (err)
		return err;

	return pagecache_write(inode, &desc_size_disk, sizeof(desc_size_disk),
			       desc_size_pos);
}

static int ext4_end_enable_verity(struct file *filp, const void *desc,
				  size_t desc_size, u64 merkle_tree_size)
{
	struct inode *inode = file_inode(filp);
	const int credits = 2; /* superblock and inode for ext4_orphan_del() */
	handle_t *handle;
	int err = 0;
	int err2;

	if (desc != NULL) {
		/* Succeeded; write the verity descriptor. */
		err = ext4_write_verity_descriptor(inode, desc, desc_size,
						   merkle_tree_size);

		/* Write all pages before clearing VERITY_IN_PROGRESS. */
		if (!err)
			err = filemap_write_and_wait(inode->i_mapping);
	}

	/* If we failed, truncate anything we wrote past i_size. */
	if (desc == NULL || err)
		ext4_truncate(inode);

	/*
	 * We must always clean up by clearing EXT4_STATE_VERITY_IN_PROGRESS and
	 * deleting the inode from the orphan list, even if something failed.
	 * If everything succeeded, we'll also set the verity bit in the same
	 * transaction.
	 */

	ext4_clear_inode_state(inode, EXT4_STATE_VERITY_IN_PROGRESS);

	handle = ext4_journal_start(inode, credits);
	if (IS_ERR(handle)) {
		ext4_orphan_del(NULL, inode);
		return PTR_ERR(handle);
	}

	err2 = ext4_orphan_del(handle, inode);
	if (err2)
		goto out_stop;

	if (desc != NULL && !err) {
		struct ext4_iloc iloc;

		err = ext4_reserve_inode_write(handle, inode, &iloc);
		if (err)
			goto out_stop;
		ext4_set_inode_flag(inode, EXT4_INODE_VERITY);
		ext4_set_inode_flags(inode);
		err = ext4_mark_iloc_dirty(handle, inode, &iloc);
	}
out_stop:
	ext4_journal_stop(handle);
	return err ?: err2;
}

static int ext4_get_verity_descriptor_location(struct inode *inode,
					       size_t *desc_size_ret,
					       u64 *desc_pos_ret)
{
	struct ext4_ext_path *path;
	struct ext4_extent *last_extent;
	u32 end_lblk;
	u64 desc_size_pos;
	__le32 desc_size_disk;
	u32 desc_size;
	u64 desc_pos;
	int err;

	/*
	 * Descriptor size is in last 4 bytes of last allocated block.
	 * See ext4_write_verity_descriptor().
	 */

	if (!ext4_test_inode_flag(inode, EXT4_INODE_EXTENTS)) {
		EXT4_ERROR_INODE(inode, "verity file doesn't use extents");
		return -EIO;
	}

	path = ext4_ext_find_extent(inode, EXT_MAX_BLOCKS - 1, NULL);
	if (IS_ERR(path))
		return PTR_ERR(path);

	last_extent = path[path->p_depth].p_ext;
	if (!last_extent) {
		EXT4_ERROR_INODE(inode, "verity file has no extents");
		ext4_ext_drop_refs(path);
		kfree(path);
		return -EIO;
	}

	end_lblk = le32_to_cpu(last_extent->ee_block) +
		   ext4_ext_get_actual_len(last_extent);
	desc_size_pos = (u64)end_lblk << inode->i_blkbits;
	ext4_ext_drop_refs(path);
	kfree(path);

	if (desc_size_pos < sizeof(desc_size_disk))
		goto bad;
	desc_size_pos -= sizeof(desc_size_disk);

	err = pagecache_read(inode, &desc_size_disk, sizeof(desc_size_disk),
			     desc_size_pos);
	if (err)
		return err;
	desc_size = le32_to_cpu(desc_size_disk);

	/*
	 * The descriptor is stored just before the desc_size_disk, but starting
	 * on a filesystem block boundary.
	 */

	if (desc_size > INT_MAX || desc_size > desc_size_pos)
		goto bad;

	desc_pos = round_down(desc_size_pos - desc_size, inode->i_sb->s_blocksize);
	if (desc_pos < ext4_verity_metadata_pos(inode))
		goto bad;

	*desc_size_ret = desc_size;
	*desc_pos_ret = desc_pos;
	return 0;

bad:
	EXT4_ERROR_INODE(inode, "verity file corrupted; can't find descriptor");
	return -EIO;
}

static int ext4_get_verity_descriptor(struct inode *inode, void *buf,
				      size_t buf_size)
{
	size_t desc_size = 0;
	u64 desc_pos = 0;
	int err;

	err = ext4_get_verity_descriptor_location(inode, &desc_size, &desc_pos);
	if (err)
		return err;

	if (buf_size) {
		if (desc_size > buf_size)
			return -ERANGE;
		err = pagecache_read(inode, buf, desc_size, desc_pos);
		if (err)
			return err;
	}
	return desc_size;
}

static struct page *ext4_read_merkle_tree_page(struct inode *inode,
					       pgoff_t index)
{
	index += ext4_verity_metadata_pos(inode) >> PAGE_SHIFT;

	return read_mapping_page(inode->i_mapping, index, NULL);
}

static int ext4_write_merkle_tree_block(struct inode *inode, const void *buf,
					u64 index, int log_blocksize)
{
	loff_t pos = ext4_verity_metadata_pos(inode) + (index << log_blocksize);

	return pagecache_write(inode, buf, 1 << log_blocksize, pos);
}

const struct fsverity_operations ext4_verityops = {
	.begin_enable_verity	= ext4_begin_enable_verity,
	.end_enable_verity	= ext4_end_enable_verity,
	.get_verity_descriptor	= ext4_get_verity_descriptor,
	.read_merkle_tree_page	= ext4_read_merkle_tree_page,
	.write_merkle_tree_block = ext4_write_merkle_tree_block,
};

struct ext4_verity_io {
	struct completion done;
	int err;
};

static void ext4_verity_end_io(struct bio *bio, int err)
{
	struct ext4_verity_io *io = bio->bi_private;

	io->err = err ?: (bio_flagged(bio, BIO_UPTODATE) ? 0 : -EIO);
	complete(&io->done);
}

int ext4_verity_readpage(void *data, struct page *page)
{
	struct inode *inode = page->mapping->host;
	struct buffer_head map = { .b_size = PAGE_SIZE };
	bool metadata = page->index >=
		(ext4_verity_metadata_pos(inode) >> PAGE_SHIFT);
	struct bio *bio;
	struct ext4_verity_io io;
	int err;

	if (inode->i_sb->s_blocksize != PAGE_SIZE) {
		err = -EIO;
		goto out;
	}
	/* Readahead may include the gap between EOF and the Merkle tree. */
	if (!metadata &&
	    page->index >= DIV_ROUND_UP(inode->i_size, PAGE_SIZE)) {
		zero_user(page, 0, PAGE_SIZE);
		goto success;
	}
	if (metadata) {
		struct buffer_head *bh;
		void *addr;

		bh = ext4_bread(NULL, inode, page->index, 0, &err);
		if (!bh) {
			err = err ?: -EIO;
			goto out;
		}
		addr = kmap_atomic(page);
		memcpy(addr, bh->b_data, PAGE_SIZE);
		kunmap_atomic(addr);
		brelse(bh);
		goto success;
	}

	err = ext4_get_block(inode, page->index, &map, 0);
	if (err)
		goto out;
	if (!buffer_mapped(&map) || buffer_unwritten(&map)) {
		zero_user(page, 0, PAGE_SIZE);
	} else {
		bio = bio_alloc(GFP_NOFS, 1);
		if (!bio) {
			err = -ENOMEM;
			goto out;
		}
		init_completion(&io.done);
		io.err = 0;
		bio->bi_bdev = map.b_bdev;
		bio->bi_sector = map.b_blocknr << (PAGE_SHIFT - 9);
		bio->bi_end_io = ext4_verity_end_io;
		bio->bi_private = &io;
		if (bio_add_page(bio, page, PAGE_SIZE, 0) != PAGE_SIZE) {
			bio_put(bio);
			err = -EIO;
			goto out;
		}
		submit_bio(READ, bio);
		wait_for_completion(&io.done);
		err = io.err;
		bio_put(bio);
		if (err)
			goto out;
	}
	/* The partial final block is hashed with zero padding. */
	if (page->index == inode->i_size >> PAGE_SHIFT)
		zero_user_segment(page, offset_in_page(inode->i_size), PAGE_SIZE);
	if (fsverity_active(inode) && !fsverity_verify_page(page)) {
		err = -EIO;
		goto out;
	}
success:
	SetPageUptodate(page);
	err = 0;
out:
	if (err) {
		ClearPageUptodate(page);
		SetPageError(page);
	}
	unlock_page(page);
	return err;
}
