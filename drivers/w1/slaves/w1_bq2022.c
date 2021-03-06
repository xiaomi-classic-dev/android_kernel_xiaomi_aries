/*
 * drivers/w1/slaves/w1_bq2022.c
 *
 * Copyright (C) 2012 Xiaomi, Inc.
 *
 * This file is licensed under the terms of the GNU General Public License
 * version 2. This program is licensed "as is" without any warranty of any
 * kind, whether express or implied.
 *
 */
/*
 * BQ2022a crypto descriptions
 *
 * Public key
 * A 32-byte key is used, and the key[32] = "WAXM".
 *
 * Crypto algorithm
 * CEPH_CRYPTO_AES
 *
 * The plain text
 * The size of the plain text is 48 bytes, defined as the following format:
 *	struct battery_security_data {
 *		int version;  // version = 1
 *		unsigned char recognition_num[16]; // "MIBATTERY-H435"
 *		unsigned char vendor[4]; // "SCUD" or "DESA"
 *		unsigned char ic_vendor[4]; "MZM" or "SONY"
 *		int capacity; // capacity = 2000
 *		unsigned char reserved[16]; // reserved[0] is "MAX" or "TI"
 *	};
 *
 * The content of EPROM
 * A 64-byte text is created after encrypted. The 64-byte  text is written
 * to the Page 0 and Page 1 of BQ2022A EPROM, while Page 3 and Page 4 are
 * unused and set all to 0xff by default.
 *
 * SCUD-MZM-TI
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x53, 0x43, 0x55, 0x44, 0x4d, 0x5a, 0x4d, 0x00, 0xd0, 0x07, 0x00, 0x00, 
 * 0x54, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
* 0x90, 0x62, 0x72, 0xbe, 0x1c, 0x92, 0x58, 0x37, 0x2b, 0x9b, 0x29, 0x46, 0x6b, 0x01, 0x07, 0xc5,
 * 0xdb, 0xd5, 0x4e, 0xe9, 0xd5, 0xa5, 0x43, 0x15, 0x19, 0xf2, 0x34, 0xc5, 0x1d, 0xe0, 0x1e, 0xca, 
 * 0x7c, 0x8e, 0x70, 0x57, 0x03, 0x8c, 0x3b, 0x8d, 0xe6, 0xef, 0xad, 0x18, 0x60, 0x97, 0x0e, 0xf4,
 *
 * SCUD-LG-TI
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x53, 0x43, 0x55, 0x44, 0x4c, 0x47, 0x00, 0x00, 0xd0, 0x07, 0x00, 0x00, 
 * 0x54, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
 * 0x7f, 0x50, 0xd3, 0x1d, 0x68, 0x2b, 0x6c, 0x9c, 0x91, 0xd4, 0xc3, 0xec, 0x3f, 0xed, 0x2d, 0x8a, 
 * 0xac, 0x22, 0x81, 0xb6, 0x2d, 0xa6, 0x52, 0xd9, 0x91, 0xa6, 0x37, 0xfe, 0x7b, 0xe4, 0xae, 0x1c, 
 * 0x7c, 0x44, 0x09, 0x9f, 0x7f, 0xca, 0x55, 0x89, 0x96, 0xd6, 0x5a, 0x87, 0x64, 0x68, 0xf3, 0x53,
 *
 * SCUD-SONY-TI
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x53, 0x43, 0x55, 0x44, 0x53, 0x4f, 0x4e, 0x59, 0xd0, 0x07, 0x00, 0x00, 
 * 0x54, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5,
 * 0x95, 0xf0, 0xa4, 0x6f, 0x7e, 0x63, 0xda, 0x83, 0x67, 0xf7, 0x97, 0x88, 0x58, 0x25, 0x1c, 0x47,
 * 0x2f, 0x17, 0x9b, 0xac, 0x44, 0x92, 0xe9, 0x29, 0xf0, 0xa6, 0xb9, 0xca, 0x1e, 0x37, 0x37, 0xd3, 
 * 0xdc, 0xe0, 0xe2, 0x72, 0x1b, 0xd9, 0xa7, 0x16, 0x1e, 0x7d, 0x89, 0x24, 0x11, 0x4b, 0xeb, 0xb5,
 *
 * SCUD-MZM-MAX
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x53, 0x43, 0x55, 0x44, 0x4d, 0x5a, 0x4d, 0x00, 0xd0, 0x07, 0x00, 0x00, 
 * 0x4d, 0x41, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5,
 * 0x90, 0x62, 0x72, 0xbe, 0x1c, 0x92, 0x58, 0x37, 0x2b, 0x9b, 0x29, 0x46, 0x6b, 0x01, 0x07, 0xc5,
 * 0x14, 0x5b, 0xe3, 0x2d, 0x0f, 0xed, 0xac, 0x22, 0x30, 0x43, 0xc0, 0x12, 0x56, 0x07, 0x87, 0x95,
 * 0xeb, 0xff, 0x2e, 0x98, 0x70, 0x96, 0x66, 0x96, 0x59, 0x44, 0x94, 0x6d, 0xe8, 0x37, 0x07, 0x30,
 *
 * SCUD-LG-MAX
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x53, 0x43, 0x55, 0x44, 0x4c, 0x47, 0x00, 0x00, 0xd0, 0x07, 0x00, 0x00, 
 * 0x4d, 0x41, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5,
 * 0x7f, 0x50, 0xd3, 0x1d, 0x68, 0x2b, 0x6c, 0x9c, 0x91, 0xd4, 0xc3, 0xec, 0x3f, 0xed, 0x2d, 0x8a, 
 * 0x2d, 0xea, 0xf1, 0x82, 0x61, 0x98, 0xef, 0x07, 0xe2, 0x03, 0xc5, 0x97, 0xb0, 0x16, 0x2a, 0x81, 
 * 0xae, 0x6f, 0x7a, 0xae, 0x47, 0x52, 0xff, 0x66, 0x07, 0x4b, 0x41, 0xe8, 0x3f, 0x2a, 0xee, 0x74,
 *
 * SCUD-SONY-MAX
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x53, 0x43, 0x55, 0x44, 0x53, 0x4f, 0x4e, 0x59, 0xd0, 0x07, 0x00, 0x00, 
 * 0x4d, 0x41, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
 * 0x95, 0xf0, 0xa4, 0x6f, 0x7e, 0x63, 0xda, 0x83, 0x67, 0xf7, 0x97, 0x88, 0x58, 0x25, 0x1c, 0x47, 
 * 0xa5, 0x57, 0x63, 0x3b, 0x21, 0x13, 0x61, 0x88, 0x13, 0xe0, 0xb5, 0x75, 0x4b, 0x4e, 0x63, 0x12, 
 * 0x06, 0xc8, 0xbf, 0x0d, 0x78, 0xc6, 0xec, 0x40, 0x42, 0xf7, 0x9d, 0xad, 0x95, 0xb2, 0x79, 0x69,
 *
 * =====================
 * DESA-MZM-TI
 * 0x33, 0x35, 0x00, 0x00, 0x44, 0x45, 0x53, 0x41, 0x4d, 0x5a, 0x4d, 0x00, 0xd0, 0x07, 0x00, 0x00, 
 * 0x54, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
 * 0x07, 0xb5, 0x67, 0x3c, 0xa2, 0xa6, 0x65, 0xde, 0x23, 0x06, 0x9b, 0x1b, 0x70, 0x87, 0xc8, 0x48, 
 * 0x2a, 0xc1, 0xeb, 0x8a, 0xca, 0x73, 0x4d, 0x68, 0x71, 0xa5, 0xea, 0x96, 0xa7, 0x8d, 0x09, 0xff, 
 * 0xc9, 0x0c, 0xa1, 0x0d, 0x06, 0x4d, 0x89, 0x83, 0xf8, 0xf5, 0xfb, 0x10, 0x1d, 0x73, 0x2b, 0x25,
 *
 * DESA-LG-TI
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x44, 0x45, 0x53, 0x41, 0x4c, 0x47, 0x00, 0x00, 0xd0, 0x07, 0x00, 0x00, 
 * 0x54, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
 * 0x38, 0xf5, 0x74, 0x87, 0x13, 0xc5, 0x60, 0x76, 0xf2, 0xf9, 0x56, 0xe8, 0x4d, 0xbd, 0xcc, 0xbb, 
 * 0x26, 0x6a, 0x85, 0x6d, 0x3a, 0x37, 0x5c, 0x61, 0xb0, 0x91, 0xdf, 0x34, 0x9e, 0xb2, 0x3e, 0x49, 
 * 0x26, 0xae, 0x7a, 0x82, 0x4b, 0xc7, 0x56, 0xe9, 0x0c, 0xbb, 0x16, 0x28, 0x87, 0x97, 0x4c, 0x68,
 *
 * DESA-SONY-TI
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x44, 0x45, 0x53, 0x41, 0x53, 0x4f, 0x4e, 0x59, 0xd0, 0x07, 0x00, 0x00, 
 * 0x54, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
 * 0xc3, 0xad, 0x9c, 0x04, 0xe7, 0x49, 0xa4, 0x66, 0x15, 0xd8, 0x78, 0x1b, 0x83, 0x3d, 0x11, 0x39, 
 * 0x86, 0x7e, 0xb1, 0x7a, 0xdb, 0xed, 0x6d, 0x9f, 0xdf, 0xac, 0x7e, 0xc0, 0x51, 0xfe, 0x82, 0x91, 
 * 0xaa, 0x61, 0x4c, 0x44, 0x2a, 0xeb, 0x54, 0xf4, 0x8f, 0x15, 0x43, 0xa7, 0xd7, 0x82, 0xb9, 0x58,
 *
 * DESA-MZM-MAX
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x44, 0x45, 0x53, 0x41, 0x4d, 0x5a, 0x4d, 0x00, 0xd0, 0x07, 0x00, 0x00, 
 * 0x4d, 0x41, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
 * 0x07, 0xb5, 0x67, 0x3c, 0xa2, 0xa6, 0x65, 0xde, 0x23, 0x06, 0x9b, 0x1b, 0x70, 0x87, 0xc8, 0x48, 
 * 0x2b, 0xaf, 0xf5, 0x6e, 0xae, 0xe7, 0x76, 0x27, 0xf7, 0xd0, 0xbb, 0x93, 0x82, 0x19, 0x11, 0x38, 
 * 0x39, 0xc8, 0x0d, 0x6a, 0xbb, 0x2b, 0x01, 0xca, 0x22, 0x8f, 0xe3, 0x8a, 0x77, 0x7e, 0x72, 0x46,
 *
 * DESA-LG-MAX
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x44, 0x45, 0x53, 0x41, 0x4c, 0x47, 0x00, 0x00, 0xd0, 0x07, 0x00, 0x00, 
 * 0x4d, 0x41, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
 * 0x38, 0xf5, 0x74, 0x87, 0x13, 0xc5, 0x60, 0x76, 0xf2, 0xf9, 0x56, 0xe8, 0x4d, 0xbd, 0xcc, 0xbb, 
 * 0x17, 0x36, 0x80, 0x43, 0x63, 0x75, 0x96, 0x8a, 0x4c, 0xba, 0x3f, 0x3d, 0xb9, 0x51, 0x0c, 0xb5, 
 * 0xc1, 0xa3, 0xee, 0xf4, 0xdd, 0x6a, 0xf6, 0x7c, 0xc3, 0xd6, 0x19, 0xde, 0x42, 0xcc, 0x98, 0x65,
 *
 * DESA-SONY-MAX
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41, 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34, 
 * 0x33, 0x35, 0x00, 0x00, 0x44, 0x45, 0x53, 0x41, 0x53, 0x4f, 0x4e, 0x59, 0xd0, 0x07, 0x00, 0x00, 
 * 0x4d, 0x41, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 *
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e, 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5, 
 * 0xc3, 0xad, 0x9c, 0x04, 0xe7, 0x49, 0xa4, 0x66, 0x15, 0xd8, 0x78, 0x1b, 0x83, 0x3d, 0x11, 0x39, 
 * 0x11, 0x10, 0x2c, 0x8a, 0xfb, 0x88, 0xe8, 0x7c, 0x27, 0x9e, 0x2e, 0x83, 0x63, 0x09, 0x8f, 0x7c, 
 * 0x2f, 0xb6, 0x44, 0x33, 0xd7, 0x24, 0xd7, 0xd7, 0xb5, 0xe3, 0xfb, 0x9a, 0x2e, 0x4e, 0x8e, 0x82,
 *
 * Below data was used in old version and is deprecated.
 * SCUD - MZM
 * 48-byte plain text:
 * 0x01, 0x00, 0x00, 0x00, 0x4d, 0x49, 0x42, 0x41,
 * 0x54, 0x54, 0x45, 0x52, 0x59, 0x2d, 0x48, 0x34,
 * 0x33, 0x35, 0x00, 0x00, 0x53, 0x43, 0x55, 0x44,
 * 0x4d, 0x5a, 0x4d, 0x00, 0xd0, 0x07, 0x00, 0x00,
 * 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 * 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
 * 64-byte encrypted text:
 * 0xed, 0x21, 0x4c, 0xe5, 0xed, 0xa9, 0x4b, 0x2e,
 * 0x62, 0xd4, 0xc6, 0xbe, 0x5e, 0xc8, 0xb6, 0xf5,
 * 0x90, 0x62, 0x72, 0xbe, 0x1c, 0x92, 0x58, 0x37,
 * 0x2b, 0x9b, 0x29, 0x46, 0x6b, 0x01, 0x07, 0xc5,
 * 0x3c, 0x78, 0x88, 0x50, 0xe9, 0x9b, 0x63, 0x40,
 * 0x26, 0x03, 0x9a, 0x55, 0xf7, 0xb9, 0x7f, 0x04,
 * 0x4a, 0x41, 0x83, 0x5d, 0x86, 0x3a, 0xe8, 0xf3,
 * 0x17, 0x6c, 0x35, 0x8e, 0x11, 0x94, 0x13, 0x10,
 *
 * DESA - MZM
 * 48-byte plain text:
 * 01 00 00 00 4d 49 42 41 54 54 45 52 59 2d 48 34
 * 33 35 00 00 44 45 53 41 4d 5a 4d 00 d0 07 00 00
 * 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 * 64-byte encrypted text:
 * ed 21 4c e5 ed a9 4b 2e 62 d4 c6 be 5e c8 b6 f5
 * 07 b5 67 3c a2 a6 65 de 23 06 9b 1b 70 87 c8 48
 * 90 4f bf 5f 2d 96 94 26 da fe 95 51 37 0e 60 c2
 * c4 ab a1 08 67 c5 9d e4 fb e2 95 d8 6e fe 5d 55
 *
 * DESA - SONY
 * 48-byte plain text:
 * 01 00 00 00 4d 49 42 41 54 54 45 52 59 2d 48 34
 * 33 35 00 00 44 45 53 41 53 4f 4e 59 d0 07 00 00
 * 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 * 64-byte encrypted test:
 * ed 21 4c e5 ed a9 4b 2e 62 d4 c6 be 5e c8 b6 f5
 * c3 ad 9c 04 e7 49 a4 66 15 d8 78 1b 83 3d 11 39
 * 86 21 c1 4c bd eb 11 49 78 af 0d f3 0a 4d c0 98
 * 0d e9 fe de c9 d2 9a 8a 98 17 69 51 e9 86 b1 86
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/device.h>
#include <linux/types.h>
#include <linux/platform_device.h>
#include <linux/mutex.h>

#include "../w1.h"
#include "../w1_int.h"
#include "../w1_family.h"

#define HDQ_CMD_SKIP_ROM (0xCC)
#define HDQ_CMD_READ_FIELD (0xF0)

#define CRYPT_COMMON_HEADER	(0xE54C21ED)
/* SCUD manufacturer */
#define CRYPT_SCUD_MZM_TI	(0xF40E9760)
#define CRYPT_SCUD_LG_TI	(0x53F36864)
#define CRYPT_SCUD_SONY_TI	(0xB5EB4B11)
#define CRYPT_SCUD_MZM_MAX	(0x300737E8)
#define CRYPT_SCUD_LG_MAX	(0x74EE2A3F)
#define CRYPT_SCUD_SONY_MAX	(0x6979B295)
/* Desay manufacturer */
#define CRYPT_DESA_MZM_TI	(0x252B731D)
#define CRYPT_DESA_LG_TI	(0x684C9787)
#define CRYPT_DESA_SONY_TI	(0x58B982D7)
#define CRYPT_DESA_MZM_MAX	(0x46727E77)
#define CRYPT_DESA_LG_MAX	(0x6598CC42)
#define CRYPT_DESA_SONY_MAX	(0x828E4E2E)

static int F_ID = 0x9;

static char batt_crypt_info[128];
static struct w1_slave *bq2022_slave;

static int w1_bq2022_read(void);

static int bq2022_debug;
static int set_debug_status_param(const char *val, struct kernel_param *kp)
{
        int ret;

        ret = param_set_int(val, kp);
        if (ret) {
                pr_err("error setting value %d\n", ret);
                return ret;
        }

        pr_info("Set debug param to %d\n", bq2022_debug);
        if (bq2022_debug) {
		int i, j;

		w1_bq2022_read();
		for (i = 0; i < 4; i++) {
			printk("Page %d ", i);
			for (j = 0; j < 32; j++) {
				printk("%02x ", *(batt_crypt_info + (i * 32 + j)));
			}
			printk("\n");
		}
	}

	return 0;
}
module_param_call(debug, set_debug_status_param, param_get_uint,
                &bq2022_debug, 0644);

int w1_bq2022_battery_id(void)
{
	int *ps, *pd;
	int ret = -1;

	/* The first four bytes */
	ps = (int *)&batt_crypt_info[0];
	/* The last four bytes */
	pd = (int *)&batt_crypt_info[60];

	if (*ps != CRYPT_COMMON_HEADER)
		return ret;

	switch(*pd) {
	/* SCUD manufacturer*/
	/* old data */
	case 0x10139411:
	case CRYPT_SCUD_MZM_TI:
		pr_info("SCUD samsung TI IC\n");
		ret = 0x1000;
		break;
	case CRYPT_SCUD_LG_TI:
		pr_info("SCUD LG TI IC\n");
		ret = 0x2000;
		break;
	case CRYPT_SCUD_SONY_TI:
		pr_info("SCUD SONY TI IC\n");
		ret = 0x1500;
		break;
	case CRYPT_SCUD_MZM_MAX:
		pr_info("SCUD samsung MAX IC\n");
		ret = 0x1000;
		break;
	case CRYPT_SCUD_LG_MAX:
		pr_info("SCUD LG MAX IC\n");
		ret = 0x2000;
		break;
	case CRYPT_SCUD_SONY_MAX:
		pr_info("SCUD SONY MAX IC\n");
		ret = 0x1500;
		break;
	/* Desay manufacturer*/
	/* old data */
	case 0x555dfe6e:
	case CRYPT_DESA_MZM_TI:
		pr_info("DESA samsung TI IC\n");
		ret = 0x1000;
		break;
	case CRYPT_DESA_LG_TI:
		pr_info("DESA LG TI IC\n");
		ret = 0x2000;
		break;
	case CRYPT_DESA_SONY_TI:
		pr_info("DESA SONY TI IC\n");
		ret = 0x1500;
		break;
	case CRYPT_DESA_MZM_MAX:
		pr_info("DESA samsung MAX IC\n");
		ret = 0x1000;
		break;
	case CRYPT_DESA_LG_MAX:
		pr_info("DESA LG MAX IC\n");
		ret = 0x2000;
		break;
	/* old data */
	case 0x86b186e9:
	case CRYPT_DESA_SONY_MAX:
		pr_info("DESA SONY MAX IC\n");
		ret = 0x1500;
		break;
	}

	return ret;
}

static int w1_bq2022_read(void)
{
	struct w1_slave *sl = bq2022_slave;
	char cmd[4];
	u8 crc, calc_crc;
	int retries = 5;

	if (!sl) {
		pr_err("No w1 device\n");
		return -1;
	}

retry:
	/* Initialization, master's mutex should be hold */
	if (!(retries--)) {
		return -1;
	}

	if (w1_reset_bus(sl->master)) {
		pr_warn("reset bus failed, just retry!\n");
		goto retry;
	}

	/* rom comm byte + read comm byte + addr 2 bytes */
	cmd[0] = HDQ_CMD_SKIP_ROM;
	cmd[1] = HDQ_CMD_READ_FIELD;
	cmd[2] = 0x0;
	cmd[3] = 0x0;

	/* send command */
	w1_write_block(sl->master, cmd, 4);

	/* crc verified for read comm byte and addr 2 bytes*/
	crc = w1_read_8(sl->master);
	calc_crc = w1_calc_crc8(&cmd[1], 3);
	if (calc_crc != crc) {
		pr_err("com crc err\n");
		goto retry;
	}

	/* read the whole memory, 1024-bit */
	w1_read_block(sl->master, batt_crypt_info, 128);

	/* crc verified for data */
	crc = w1_read_8(sl->master);
	calc_crc = w1_calc_crc8(batt_crypt_info, 128);
	if (calc_crc != crc) {
		pr_err("data crc err\n");
		goto retry;
	}

	return 0;

}

static int w1_bq2022_add_slave(struct w1_slave *sl)
{
	bq2022_slave = sl;
	return w1_bq2022_read();
}

static void w1_bq2022_remove_slave(struct w1_slave *sl)
{
	bq2022_slave = NULL;
}

static struct w1_family_ops w1_bq2022_fops = {
	.add_slave	= w1_bq2022_add_slave,
	.remove_slave	= w1_bq2022_remove_slave,
};

static struct w1_family w1_bq2022_family = {
	.fid = 1,
	.fops = &w1_bq2022_fops,
};

static int __init w1_bq2022_init(void)
{
	if (F_ID)
		w1_bq2022_family.fid = F_ID;

	return w1_register_family(&w1_bq2022_family);
}

static void __exit w1_bq2022_exit(void)
{
	w1_unregister_family(&w1_bq2022_family);
}

module_init(w1_bq2022_init);
module_exit(w1_bq2022_exit);

module_param(F_ID, int, S_IRUSR);
MODULE_PARM_DESC(F_ID, "1-wire slave FID for BQ device");

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Xiaomi Ltd");
MODULE_DESCRIPTION("HDQ/1-wire slave driver bq2022 battery chip");
