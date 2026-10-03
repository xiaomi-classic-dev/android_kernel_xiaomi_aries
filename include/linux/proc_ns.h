#ifndef _LINUX_PROC_NS_H
#define _LINUX_PROC_NS_H
#include <linux/proc_fs.h>
#include <linux/ns_common.h>
#define PROC_CGROUP_INIT_INO 0xEFFFFFFBU
extern const struct proc_ns_operations cgroupns_operations;
#define ns_alloc_inum(ns) proc_alloc_inum(&(ns)->inum)
#define ns_free_inum(ns) proc_free_inum((ns)->inum)
#endif
