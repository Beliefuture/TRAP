SELECT ps_suppkey, l_suppkey, o_orderstatus FROM orders JOIN lineitem ON l_orderkey = o_orderkey JOIN partsupp ON ps_suppkey = l_suppkey WHERE ps_partkey >= 55775 AND o_custkey > 122389 AND l_extendedprice > 93257.76 AND l_quantity < 40.0 AND l_shipinstruct < 'COLLECT COD' AND l_receiptdate >= '1993-11-01';
SELECT l_partkey, o_orderpriority FROM lineitem, orders WHERE l_discount > 0.08 AND l_comment = 'ly even packages af' AND l_suppkey > 5532 AND o_custkey < 4730;
SELECT l_linestatus, o_totalprice, ps_partkey FROM orders JOIN lineitem ON l_orderkey = o_orderkey JOIN partsupp ON ps_suppkey = l_suppkey WHERE l_linestatus <> 'F' AND l_receiptdate > '1993-06-28' AND l_partkey = 17221 AND l_commitdate = '1994-03-05' AND l_comment <> 'furiously unusual theodolites wake';
SELECT s_address, l_comment, n_comment, ps_suppkey FROM nation JOIN supplier ON s_nationkey = n_nationkey JOIN partsupp ON ps_suppkey = s_suppkey JOIN lineitem ON l_suppkey = ps_suppkey WHERE l_shipinstruct > 'TAKE BACK RETURN' AND ps_partkey <> 49182 AND l_comment > 'odolites along the blithely special a';
SELECT l_quantity, ps_supplycost, MIN(ps_supplycost) FROM partsupp JOIN lineitem ON l_suppkey = ps_suppkey WHERE l_shipdate <= '1997-09-11' AND l_commitdate = '1996-01-02' AND ps_supplycost < 628.53 AND l_extendedprice <> 27509.72 GROUP BY l_quantity, ps_supplycost ORDER BY l_quantity ASC;
SELECT c_acctbal, s_address, l_suppkey, o_orderdate, ps_partkey FROM customer JOIN orders ON o_custkey = c_custkey JOIN lineitem ON l_orderkey = o_orderkey JOIN partsupp ON ps_suppkey = l_suppkey JOIN supplier ON s_suppkey = ps_suppkey WHERE s_phone < '17-292-821-2297' AND ps_suppkey = 6420 AND s_comment >= 'e carefully around the excuse';
SELECT l_linenumber, o_orderkey FROM orders JOIN lineitem ON l_orderkey = o_orderkey WHERE o_orderstatus > 'F' AND l_suppkey = 5491 AND o_custkey <> 140612 AND l_shipinstruct <= 'COLLECT COD';
SELECT l_orderkey, ps_partkey, COUNT(ps_suppkey), AVG(l_quantity), AVG(ps_partkey) FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey WHERE ps_partkey <= 173618 AND l_linestatus > 'O' AND l_returnflag = 'R' GROUP BY l_orderkey, ps_partkey HAVING COUNT(ps_suppkey) <> 2074 AND AVG(l_quantity) < 4 AND AVG(ps_partkey) <> 161;
SELECT l_partkey, ps_partkey, MAX(l_receiptdate) FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey WHERE l_discount = 0.01 AND l_shipdate = '1993-01-26' AND l_extendedprice <= 41183.22 GROUP BY l_partkey, ps_partkey ORDER BY MAX(l_receiptdate) ASC;
SELECT l_receiptdate FROM lineitem WHERE l_quantity = 32.0 AND l_commitdate = '1993-08-11' AND l_suppkey <> 8624 AND l_partkey <> 25902 ORDER BY l_receiptdate ASC;
SELECT l_discount, o_custkey, n_name, c_comment, COUNT(o_orderkey) FROM nation JOIN customer ON c_nationkey = n_nationkey JOIN orders ON o_custkey = c_custkey JOIN lineitem ON l_orderkey = o_orderkey WHERE o_orderpriority > '5-LOW' AND o_custkey <> 59644 GROUP BY l_discount, o_custkey, n_name, c_comment ORDER BY l_discount DESC;
SELECT ps_availqty, s_address FROM supplier, partsupp WHERE s_acctbal <= 6938.43 AND ps_comment = 'accounts haggle slyly about the quickly special packages. quickly furious requests are according to' AND s_address < 'rCuPMo62kci' AND s_name > 'Supplier#000000551';
SELECT ps_availqty FROM partsupp WHERE ps_availqty = 5628 AND ps_supplycost <> 164.19 AND ps_comment <= 'uffily express instructions. carefully special theodolites are against the final packages. furiously ironic pains sleep quickly about' AND ps_partkey >= 4732 ORDER BY ps_availqty DESC;
SELECT c_phone, n_regionkey, ps_partkey, p_partkey, s_address FROM customer JOIN nation ON n_nationkey = c_nationkey JOIN supplier ON s_nationkey = n_nationkey JOIN partsupp ON ps_suppkey = s_suppkey JOIN part ON p_partkey = ps_partkey WHERE c_nationkey < 12 AND c_phone = '17-265-877-1490' ORDER BY s_address DESC, c_phone ASC, ps_partkey DESC;
SELECT n_comment, s_suppkey, ps_comment FROM nation JOIN supplier ON s_nationkey = n_nationkey JOIN partsupp ON ps_suppkey = s_suppkey WHERE n_name = 'JAPAN' AND ps_availqty < 1842 AND s_nationkey <= 10 AND ps_comment > 'wake carefully silently final packages. carefully even deposits use special, re' AND s_address < 'O1KJE67Z,KykRf8mV72VTnDG35PhSR0S0CJlYFi7' AND ps_supplycost < 892.65 ORDER BY ps_comment ASC, n_comment ASC, s_suppkey DESC;
SELECT o_clerk, c_comment FROM customer, orders WHERE o_shippriority >= 0 AND c_address < 'p4U,vB,Jz3SkV9tKHTOlNgDJ' AND c_comment > 'bove the express, final deposits wake furiously furiou' AND c_phone = '15-411-560-1974' AND o_orderkey = 2232065;
SELECT ps_supplycost, n_nationkey, c_mktsegment, l_partkey, r_comment, o_orderdate FROM region JOIN nation ON n_regionkey = r_regionkey JOIN customer ON c_nationkey = n_nationkey JOIN orders ON o_custkey = c_custkey JOIN lineitem ON l_orderkey = o_orderkey JOIN partsupp ON ps_suppkey = l_suppkey WHERE l_extendedprice = 15675.48 AND n_regionkey = 2;
SELECT c_acctbal, l_commitdate, o_orderstatus, ps_suppkey FROM partsupp JOIN lineitem ON l_suppkey = ps_suppkey JOIN orders ON o_orderkey = l_orderkey JOIN customer ON c_custkey = o_custkey WHERE l_receiptdate = '1998-06-30' AND l_shipinstruct > 'TAKE BACK RETURN' AND c_phone >= '19-646-636-2249' ORDER BY o_orderstatus DESC, ps_suppkey DESC, l_commitdate DESC, c_acctbal DESC;
SELECT l_extendedprice, s_phone, ps_availqty FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN supplier ON s_suppkey = ps_suppkey WHERE s_comment = 'requests. regular, regular accounts detect quickly bold accounts. enticingly final packages' AND ps_availqty < 2643 AND l_partkey <> 96753 ORDER BY s_phone DESC, l_extendedprice ASC, ps_availqty DESC;
SELECT s_acctbal, ps_supplycost FROM supplier, partsupp WHERE s_suppkey <= 1558 AND ps_partkey < 64064 AND ps_suppkey <> 2099 AND ps_supplycost = 169.47;
SELECT l_shipdate, ps_suppkey, p_type FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN part ON p_partkey = ps_partkey WHERE l_tax <> 0.0 AND p_partkey = 173355 AND ps_availqty <= 2174 AND l_orderkey >= 2557408 ORDER BY p_type ASC, l_shipdate DESC, ps_suppkey DESC;
SELECT l_suppkey, o_clerk, MIN(l_shipinstruct) FROM orders JOIN lineitem ON l_orderkey = o_orderkey WHERE o_clerk = 'Clerk#000000934' AND l_partkey >= 147429 AND o_orderkey >= 3241411 GROUP BY l_suppkey, o_clerk HAVING MIN(l_shipinstruct) <= 'COLLECT COD' ORDER BY MIN(l_shipinstruct) ASC;
SELECT l_linenumber, ps_suppkey, p_type FROM part JOIN partsupp ON ps_partkey = p_partkey JOIN lineitem ON l_suppkey = ps_suppkey WHERE p_mfgr < 'Manufacturer#3' AND l_comment = 'blithely regular grouches slee' AND l_extendedprice = 98584.08 AND l_linenumber <> 2;
SELECT ps_comment, l_partkey, p_container FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN part ON p_partkey = ps_partkey WHERE ps_suppkey = 6387 AND l_shipinstruct >= 'NONE' AND l_quantity > 23.0 ORDER BY l_partkey DESC;
SELECT p_type, s_name, ps_comment FROM part, partsupp, supplier WHERE p_comment >= 'rts sleep according t' AND ps_availqty = 4463 AND p_partkey < 181505 AND p_type = 'STANDARD BRUSHED STEEL' AND s_address > ',eN75B9Wo,VoklFVVnt4' ORDER BY s_name DESC;
SELECT o_orderpriority, c_address, MAX(c_acctbal) FROM customer, orders WHERE o_orderpriority > '5-LOW' AND c_acctbal <> 7232.97 AND c_phone < '21-230-264-1156' GROUP BY o_orderpriority, c_address ORDER BY c_address DESC, o_orderpriority DESC;
SELECT l_shipmode, ps_partkey, s_nationkey FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN supplier ON s_suppkey = ps_suppkey WHERE l_partkey = 63858 AND l_commitdate <= '1993-04-29' AND s_address >= 'NxR4B,oub4GdMpx8lVaR' AND ps_supplycost < 343.83 ORDER BY s_nationkey ASC, ps_partkey ASC;
SELECT s_phone, l_returnflag, ps_availqty FROM supplier, partsupp, lineitem WHERE l_comment = 'lar foxes according to the' AND ps_partkey = 188984 AND l_returnflag = 'N' AND l_shipmode <= 'TRUCK';
SELECT s_suppkey, c_name, n_name, o_custkey FROM supplier JOIN nation ON n_nationkey = s_nationkey JOIN customer ON c_nationkey = n_nationkey JOIN orders ON o_custkey = c_custkey WHERE c_address >= '6qBCAyJgnZeUIE5e9h' AND s_nationkey < 23 AND o_comment = 'fily slyly bold instructions. b' ORDER BY n_name DESC;
SELECT o_custkey, r_regionkey, c_mktsegment, n_comment FROM orders JOIN customer ON c_custkey = o_custkey JOIN nation ON n_nationkey = c_nationkey JOIN region ON r_regionkey = n_regionkey WHERE r_comment >= 'lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to' AND o_orderdate = '1998-06-21' AND c_custkey > 91366 ORDER BY c_mktsegment ASC, o_custkey ASC, n_comment DESC;
SELECT l_comment FROM lineitem WHERE l_comment <> 's wake across the blithely ev' AND l_suppkey < 701 AND l_shipinstruct >= 'TAKE BACK RETURN' AND l_orderkey > 4853414 ORDER BY l_comment DESC;
SELECT s_nationkey, ps_supplycost, n_name, r_comment FROM region JOIN nation ON n_regionkey = r_regionkey JOIN supplier ON s_nationkey = n_nationkey JOIN partsupp ON ps_suppkey = s_suppkey WHERE s_phone > '27-187-885-5530' AND n_comment <= 'ic deposits are blithely about the carefully regular pa' AND r_name <= 'AMERICA';
SELECT o_shippriority, c_phone FROM orders, customer WHERE c_phone > '20-254-729-7009' AND c_address >= 'AWmzgVcPqQmVB2lZbwTvU4BcKhNdzk' AND c_comment <> 'osits according to the furiously unusual pinto beans x-ray slyly according to t' AND o_totalprice = 152398.31;
SELECT o_totalprice, l_tax, s_suppkey, ps_partkey FROM supplier JOIN partsupp ON ps_suppkey = s_suppkey JOIN lineitem ON l_suppkey = ps_suppkey JOIN orders ON o_orderkey = l_orderkey WHERE ps_partkey <= 188984 AND l_discount = 0.06 AND s_address = 'DeCSqduLu2JE9brcwCj';
SELECT l_receiptdate, o_orderdate, n_comment, r_comment, ps_supplycost, c_comment FROM partsupp JOIN lineitem ON l_suppkey = ps_suppkey JOIN orders ON o_orderkey = l_orderkey JOIN customer ON c_custkey = o_custkey JOIN nation ON n_nationkey = c_nationkey JOIN region ON r_regionkey = n_regionkey WHERE r_name < 'AFRICA' AND l_shipmode > 'TRUCK';
SELECT o_orderstatus, l_orderkey, ps_supplycost FROM partsupp JOIN lineitem ON l_suppkey = ps_suppkey JOIN orders ON o_orderkey = l_orderkey WHERE l_extendedprice > 31446.03 AND l_partkey <> 152696 AND l_linestatus < 'F' ORDER BY l_orderkey ASC, ps_supplycost ASC, o_orderstatus DESC;
SELECT n_name, c_address, o_orderstatus FROM nation, customer, orders WHERE c_address <= 'dfsCDFGpDdYSjLHIcaO2 X9W8YwYLB7XlF' AND c_name = 'Customer#000131107' AND o_orderstatus > 'P' ORDER BY n_name DESC;
SELECT l_shipdate, o_custkey, c_address, n_nationkey FROM lineitem JOIN orders ON o_orderkey = l_orderkey JOIN customer ON c_custkey = o_custkey JOIN nation ON n_nationkey = c_nationkey WHERE c_comment = 'nod carefully against the regular requests. ironic instructions affix fluffily afte' AND c_address <> 'gzNtUmM66Zw' AND l_partkey < 76451 ORDER BY c_address ASC, o_custkey ASC, n_nationkey ASC, l_shipdate ASC;
SELECT o_orderdate, c_phone FROM customer, orders WHERE o_orderpriority = '3-MEDIUM' AND o_orderstatus <= 'O' AND c_custkey > 88955 AND c_mktsegment = 'FURNITURE' AND c_comment > 'ts sleep carefully fluffily final warthogs. regul' AND c_name <> 'Customer#000122315' AND o_orderkey <> 2043137 ORDER BY o_orderdate DESC;
SELECT o_clerk FROM orders WHERE o_orderdate <> '1998-04-15' AND o_custkey <= 134716 AND o_totalprice <= 133709.48 AND o_shippriority > 0 AND o_comment > 'e carefully after the regular instructions-- furiously final instructi' AND o_orderkey <> 4716610 AND o_clerk <> 'Clerk#000000721' AND o_orderstatus <= 'F' AND o_orderpriority <= '5-LOW';
SELECT s_name, ps_suppkey FROM supplier, partsupp WHERE s_address = 'Cs93kCGRA6HlNVZjgrU,5Fi 1F3 vx' AND ps_comment <= 'ular requests boost pinto beans. even, regular deposits cajole according to the regular foxes. even deposits according to the bravely daring requests print caref' AND ps_supplycost = 456.0 AND s_phone < '16-542-608-8183' AND ps_partkey > 152695 ORDER BY s_name ASC;
SELECT o_orderkey, c_name FROM orders, customer WHERE o_orderstatus > 'P' AND o_orderpriority >= '4-NOT SPECIFIED' AND o_shippriority = 0 AND o_totalprice > 306820.41;
SELECT ps_availqty, p_mfgr FROM part, partsupp WHERE p_comment > 'ronic pinto bea' AND ps_partkey <= 108173 AND ps_supplycost > 892.65 AND p_size > 28 ORDER BY p_mfgr DESC;
SELECT l_extendedprice, ps_supplycost, s_nationkey, n_regionkey FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN supplier ON s_suppkey = ps_suppkey JOIN nation ON n_nationkey = s_nationkey WHERE l_comment < 'o the slyly even f' AND l_receiptdate <> '1992-09-16' AND l_commitdate = '1998-01-07';
SELECT o_orderstatus, c_custkey, MAX(o_clerk) FROM customer JOIN orders ON o_custkey = c_custkey WHERE o_custkey = 115963 AND o_orderstatus <= 'F' AND c_acctbal = 2379.26 GROUP BY o_orderstatus, c_custkey ORDER BY MAX(o_clerk) ASC;
SELECT n_name, o_orderstatus, c_custkey FROM nation, customer, orders WHERE n_comment > 'nic deposits boost atop the quickly final requests? quickly regula' AND n_name < 'BRAZIL' AND o_custkey = 35401 ORDER BY o_orderstatus DESC;
SELECT s_address, l_quantity, ps_suppkey FROM lineitem, partsupp, supplier WHERE l_orderkey >= 5367938 AND l_comment = 'ular escap' AND s_phone = '24-810-371-6779' AND s_suppkey >= 8612 AND l_extendedprice = 6394.08;
SELECT n_comment, o_orderpriority, c_comment, SUM(o_totalprice), MAX(n_comment) FROM orders, customer, nation WHERE n_regionkey <= 2 AND o_shippriority < 0 AND c_address = 'T7atVtPnd,LBdFg8BUqYBUab' AND o_orderpriority > '3-MEDIUM' GROUP BY n_comment, o_orderpriority, c_comment;
SELECT c_address, n_regionkey, s_name FROM customer, nation, supplier WHERE n_name >= 'KENYA' AND n_comment >= 'ven packages wake quickly. regu' AND s_address < 'c6fBN9a 6EOcB1ZjbImMBAQMwI BKScDNVRP8' AND s_suppkey >= 4702 AND s_acctbal = 7720.17 ORDER BY n_regionkey DESC;
SELECT p_mfgr, ps_suppkey FROM part, partsupp WHERE p_type = 'ECONOMY PLATED BRASS' AND p_partkey <= 105242 AND ps_comment < 'lent accounts detect quickly accounts. regularly regular foxes haggle furiously. foxes use alongside of the special, pending pinto beans. deposits sleep. furiously unu' AND p_container < 'SM PACK' ORDER BY ps_suppkey ASC;
SELECT s_phone, ps_comment, l_orderkey, o_comment FROM supplier JOIN partsupp ON ps_suppkey = s_suppkey JOIN lineitem ON l_suppkey = ps_suppkey JOIN orders ON o_orderkey = l_orderkey WHERE l_commitdate > '1996-02-23' AND l_linestatus >= 'F' AND ps_comment = 'sts along the pending pinto beans affix quietly about the fluffily final pinto beans. special, bold requests may sleep slyly special requests. pinto beans sleep. deposits breach. quickly re';
SELECT o_shippriority, c_mktsegment FROM customer, orders WHERE c_mktsegment > 'HOUSEHOLD' AND o_shippriority < 0 AND o_orderkey < 4064390 AND o_comment = 'nag furiously fluffi' ORDER BY o_shippriority DESC;
SELECT c_nationkey, o_custkey, MIN(c_nationkey) FROM customer JOIN orders ON o_custkey = c_custkey WHERE o_shippriority = 0 AND o_clerk = 'Clerk#000000314' AND c_name < 'Customer#000116589' AND c_custkey <= 90696 GROUP BY c_nationkey, o_custkey ORDER BY MIN(c_nationkey) ASC;
SELECT o_custkey, c_comment FROM customer JOIN orders ON o_custkey = c_custkey WHERE c_address < 'JaLWr3HdBDxkCkb3VVxig9uP9jmboNlT9cG7DBR' AND c_custkey < 33278 AND o_orderstatus >= 'F' AND c_phone = '19-600-929-1992';
SELECT l_shipdate, s_comment, ps_availqty FROM supplier JOIN partsupp ON ps_suppkey = s_suppkey JOIN lineitem ON l_suppkey = ps_suppkey WHERE s_address <> 'WwpiochhF7rKPsIqQguH' AND l_partkey = 79739 AND l_orderkey <> 943812 AND l_discount > 0.08;
SELECT ps_supplycost, s_name FROM supplier, partsupp WHERE ps_partkey < 120117 AND s_nationkey < 14 AND ps_comment <= 'uests along the regular platelets nag furiously against the slyly silent ideas-- quickly pending foxes cajole furiously slyly even i' AND ps_supplycost = 45.64 AND s_comment > 'efully. fluffily regular packages affix regular instructions. sly, unusual deposits haggle among' AND ps_availqty <= 6331 AND s_address > 'QYA7LJ8f3qcqUW70f8x2 7nU9Xf1BRh20iV';
SELECT l_returnflag, p_mfgr, ps_supplycost, MIN(p_brand), SUM(ps_supplycost) FROM part JOIN partsupp ON ps_partkey = p_partkey JOIN lineitem ON l_suppkey = ps_suppkey WHERE ps_comment = 'thes haggle carefully. stealthy deposits cajole about the final, express dolphins.' AND p_retailprice > 1815.88 AND l_quantity < 11.0 GROUP BY l_returnflag, p_mfgr, ps_supplycost ORDER BY p_mfgr DESC, l_returnflag ASC;
SELECT ps_partkey, s_address FROM partsupp, supplier WHERE ps_suppkey = 8989 AND s_nationkey >= 5 AND s_phone < '19-740-622-6170' AND ps_comment >= 'g blithely against the packages: blithely ironic epitaphs at the platelets sleep careful' ORDER BY s_address DESC;
SELECT l_commitdate, o_shippriority, ps_supplycost, c_comment FROM customer JOIN orders ON o_custkey = c_custkey JOIN lineitem ON l_orderkey = o_orderkey JOIN partsupp ON ps_suppkey = l_suppkey WHERE o_orderdate < '1998-04-15' AND c_address = 'D8P4rDP3ndvcKHyN Rti4EOB' AND o_custkey < 4730;
SELECT o_clerk, c_custkey, n_name, MAX(c_nationkey) FROM orders, customer, nation WHERE o_totalprice <= 174152.67 AND c_name = 'Customer#000070744' AND o_comment = 'fily above the silent, even hockey players. carefully final theodolites engage' GROUP BY o_clerk, c_custkey, n_name ORDER BY MAX(c_nationkey) DESC;
SELECT c_custkey, o_shippriority FROM customer, orders WHERE c_name >= 'Customer#000070744' AND o_shippriority < 0 AND c_address <= 'D8P4rDP3ndvcKHyN Rti4EOB' AND o_comment = 'ously even theodolites. even p' AND o_orderstatus < 'P' AND o_clerk <= 'Clerk#000000818' ORDER BY c_custkey ASC;
SELECT l_commitdate, ps_suppkey FROM partsupp JOIN lineitem ON l_suppkey = ps_suppkey WHERE l_discount = 0.01 AND l_comment = 'ding packages nag. even asymptotes aro' AND l_tax > 0.0 AND ps_availqty >= 9401;
SELECT o_orderstatus, l_shipmode FROM lineitem, orders WHERE o_orderstatus > 'O' AND o_orderkey >= 4064390 AND l_linenumber > 7 AND l_tax <> 0.02;
SELECT l_returnflag, ps_comment FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey WHERE l_linenumber > 4 AND ps_supplycost = 771.41 AND ps_suppkey <= 7856 AND l_receiptdate < '1994-12-01' ORDER BY l_returnflag ASC;
SELECT l_commitdate, ps_suppkey, s_phone, AVG(ps_supplycost) FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN supplier ON s_suppkey = ps_suppkey WHERE s_address >= 'H5tDfi,XJ8BuciyUcOao1WXbXOWIGBR' AND ps_availqty = 7950 GROUP BY l_commitdate, ps_suppkey, s_phone HAVING AVG(ps_supplycost) < 160.7 ORDER BY ps_suppkey DESC, l_commitdate ASC, s_phone ASC;
SELECT o_custkey, l_discount FROM lineitem, orders WHERE l_partkey < 31126 AND l_extendedprice >= 38634.96 AND l_shipmode = 'TRUCK' AND l_tax <= 0.01 ORDER BY l_discount ASC;
SELECT l_extendedprice, ps_partkey FROM partsupp, lineitem WHERE l_returnflag > 'N' AND l_tax >= 0.02 AND l_extendedprice = 49926.12 AND l_orderkey <> 4099425 ORDER BY l_extendedprice ASC, ps_partkey ASC;
SELECT p_size, ps_supplycost FROM part JOIN partsupp ON ps_partkey = p_partkey WHERE ps_comment <> 's. carefully regular instructions hang furiously slyly pending accounts. bold packages' AND p_name > 'rosy aquamarine purple sky violet' AND ps_availqty = 9854 AND ps_partkey > 120117 ORDER BY ps_supplycost ASC;
SELECT c_address, o_clerk FROM orders JOIN customer ON c_custkey = o_custkey WHERE c_nationkey = 19 AND o_comment <= 'the unusual, pending shea' AND o_orderdate = '1997-08-21' AND c_name < 'Customer#000070744' ORDER BY o_clerk ASC;
SELECT c_name, l_receiptdate, o_custkey FROM lineitem JOIN orders ON o_orderkey = l_orderkey JOIN customer ON c_custkey = o_custkey WHERE l_receiptdate <> '1997-05-23' AND c_name >= 'Customer#000122315' AND c_address > 'ZKkTx050heGptGiWaYQikKYinHpi' AND c_mktsegment <> 'FURNITURE' AND l_quantity >= 7.0;
SELECT l_suppkey, ps_partkey, MAX(ps_suppkey) FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey WHERE l_linestatus > 'O' AND l_discount <> 0.05 AND l_comment <= 'special deposits detect s' GROUP BY l_suppkey, ps_partkey HAVING MAX(ps_suppkey) <= 5092;
SELECT ps_partkey, s_phone, p_name, n_nationkey, r_regionkey, COUNT(ps_availqty) FROM region JOIN nation ON n_regionkey = r_regionkey JOIN supplier ON s_nationkey = n_nationkey JOIN partsupp ON ps_suppkey = s_suppkey JOIN part ON p_partkey = ps_partkey WHERE n_comment = 'ss excuses cajole slyly across the packages. deposits print aroun' AND p_name > 'chartreuse grey drab honeydew seashell' GROUP BY ps_partkey, s_phone, p_name, n_nationkey, r_regionkey;
SELECT c_name, o_clerk FROM customer, orders WHERE c_mktsegment <= 'BUILDING' AND c_acctbal >= 799.99 AND o_orderkey <> 1436102 AND c_name < 'Customer#000116589' AND c_comment <> 'equests over the slyly unusual deposits' AND o_orderpriority = '4-NOT SPECIFIED' ORDER BY c_name DESC;
SELECT o_orderpriority, l_discount, p_size, ps_supplycost FROM orders JOIN lineitem ON l_orderkey = o_orderkey JOIN partsupp ON ps_suppkey = l_suppkey JOIN part ON p_partkey = ps_partkey WHERE p_retailprice > 1815.88 AND l_suppkey <= 1197 AND ps_partkey = 152695;
SELECT ps_supplycost, n_comment, s_comment FROM partsupp, supplier, nation WHERE s_acctbal <= 2543.89 AND n_nationkey < 3 AND s_phone = '34-272-359-1149' ORDER BY ps_supplycost ASC;
SELECT l_shipdate, ps_comment, o_orderpriority, COUNT(ps_suppkey) FROM partsupp JOIN lineitem ON l_suppkey = ps_suppkey JOIN orders ON o_orderkey = l_orderkey WHERE l_shipmode < 'AIR' AND l_shipinstruct >= 'COLLECT COD' AND l_extendedprice <> 5144.01 GROUP BY l_shipdate, ps_comment, o_orderpriority HAVING COUNT(ps_suppkey) < 1845 ORDER BY COUNT(ps_suppkey) ASC;
SELECT ps_comment, n_nationkey, c_acctbal, s_name FROM customer JOIN nation ON n_nationkey = c_nationkey JOIN supplier ON s_nationkey = n_nationkey JOIN partsupp ON ps_suppkey = s_suppkey WHERE s_acctbal > 4975.09 AND c_comment = 'e permanently. stealthy pinto beans haggle slyly. ironic, ironic foxe' AND n_regionkey = 0 AND c_mktsegment = 'FURNITURE' AND c_acctbal = 912.55 ORDER BY n_nationkey DESC, s_name DESC;
SELECT c_phone, o_totalprice FROM customer, orders WHERE o_orderkey < 3777890 AND o_totalprice < 306396.41 AND c_custkey >= 149303 AND o_orderdate < '1997-02-07' AND c_mktsegment > 'FURNITURE' ORDER BY o_totalprice ASC;
SELECT l_commitdate, ps_suppkey FROM partsupp JOIN lineitem ON l_suppkey = ps_suppkey WHERE l_suppkey = 7722 AND l_returnflag < 'R' AND ps_availqty = 3676 AND l_shipinstruct > 'DELIVER IN PERSON' AND l_discount = 0.01 ORDER BY l_commitdate ASC;
SELECT ps_partkey, p_brand FROM part JOIN partsupp ON ps_partkey = p_partkey WHERE p_comment > 'dolites use; furious' AND p_mfgr < 'Manufacturer#3' AND p_name = 'violet chartreuse red blue medium' AND p_brand > 'Brand#22';
SELECT l_orderkey, p_retailprice, c_phone, o_orderkey, ps_comment FROM customer JOIN orders ON o_custkey = c_custkey JOIN lineitem ON l_orderkey = o_orderkey JOIN partsupp ON ps_suppkey = l_suppkey JOIN part ON p_partkey = ps_partkey WHERE l_shipdate <= '1995-05-18' AND o_orderstatus > 'P' ORDER BY c_phone DESC, o_orderkey ASC, l_orderkey ASC, ps_comment ASC;
SELECT o_totalprice, l_orderkey, c_name FROM lineitem JOIN orders ON o_orderkey = l_orderkey JOIN customer ON c_custkey = o_custkey WHERE l_linestatus = 'O' AND o_totalprice <= 113745.35 AND c_comment = 'ously final deposits breach fluffily silent, silent' AND o_custkey <= 99901 ORDER BY l_orderkey ASC, c_name DESC, o_totalprice DESC;
SELECT ps_availqty, l_tax, p_retailprice FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN part ON p_partkey = ps_partkey WHERE l_commitdate >= '1993-10-02' AND p_brand <> 'Brand#35' AND p_name = 'chiffon metallic orange turquoise snow' ORDER BY ps_availqty ASC;
SELECT r_comment, n_regionkey, c_acctbal FROM region, nation, customer WHERE c_acctbal <= 8449.32 AND n_name <> 'JAPAN' AND r_comment > 'lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to' ORDER BY n_regionkey ASC;
SELECT o_comment, s_nationkey, c_address, n_name FROM orders JOIN customer ON c_custkey = o_custkey JOIN nation ON n_nationkey = c_nationkey JOIN supplier ON s_nationkey = n_nationkey WHERE s_address >= 'W89jjgy458' AND c_acctbal = 9221.33 AND s_phone <> '32-950-749-3092' ORDER BY n_name DESC, c_address DESC;
SELECT l_returnflag, o_comment, ps_comment FROM partsupp JOIN lineitem ON l_suppkey = ps_suppkey JOIN orders ON o_orderkey = l_orderkey WHERE l_shipmode <= 'TRUCK' AND o_clerk <= 'Clerk#000000977' AND l_linenumber <> 7 AND l_extendedprice = 39508.17 ORDER BY o_comment DESC, ps_comment ASC;
SELECT s_comment, ps_partkey FROM supplier, partsupp WHERE ps_comment = 'ular dependencies above the accounts cajole final accounts. quickly unusual pinto beans haggle fluffily. blithe' AND s_nationkey <= 15 AND s_comment >= 'fluffily furiously pending accoun' AND s_name >= 'Supplier#000001708';
SELECT s_phone, l_discount, ps_comment FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN supplier ON s_suppkey = ps_suppkey WHERE l_quantity <> 29.0 AND l_extendedprice <= 12412.64 AND l_shipdate > '1996-11-22' AND l_tax = 0.06 AND l_receiptdate = '1997-06-10' ORDER BY l_discount ASC;
SELECT o_orderstatus, c_acctbal FROM orders, customer WHERE o_totalprice > 107147.42 AND o_clerk <> 'Clerk#000000402' AND c_acctbal < 2827.03 AND o_comment > 'silent requests. regular pinto be' AND o_orderdate = '1997-03-19' AND c_comment > 'ggle carefully after the furiously regular theodolites; slyly quick requests are. carefull' ORDER BY c_acctbal DESC;
SELECT l_suppkey, r_regionkey, s_suppkey, n_name, ps_suppkey, MIN(r_comment), COUNT(ps_suppkey) FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey JOIN supplier ON s_suppkey = ps_suppkey JOIN nation ON n_nationkey = s_nationkey JOIN region ON r_regionkey = n_regionkey WHERE ps_supplycost = 558.99 AND l_suppkey >= 4942 GROUP BY l_suppkey, r_regionkey, s_suppkey, n_name, ps_suppkey;
SELECT l_shipdate, p_partkey, ps_comment FROM part JOIN partsupp ON ps_partkey = p_partkey JOIN lineitem ON l_suppkey = ps_suppkey WHERE p_mfgr <= 'Manufacturer#3' AND ps_availqty = 9934 AND p_retailprice < 1115.03 ORDER BY p_partkey DESC, ps_comment ASC;
SELECT o_comment, l_receiptdate FROM orders JOIN lineitem ON l_orderkey = o_orderkey WHERE o_shippriority < 0 AND l_receiptdate <> '1992-11-18' AND o_clerk < 'Clerk#000000433' AND l_shipmode < 'SHIP' AND o_comment = 'even dugouts are slyly' AND o_orderpriority <= '3-MEDIUM';
SELECT l_shipdate, ps_partkey FROM lineitem JOIN partsupp ON ps_suppkey = l_suppkey WHERE ps_partkey >= 138335 AND l_returnflag <> 'R' AND l_shipmode >= 'REG AIR' AND ps_availqty = 4807;
SELECT l_partkey, o_orderkey FROM orders, lineitem WHERE l_linestatus > 'O' AND l_partkey <> 120307 AND l_quantity >= 8.0 AND l_shipmode <> 'RAIL';
SELECT c_nationkey, n_name, o_clerk, r_name, COUNT(c_custkey) FROM region, nation, customer, orders WHERE c_custkey = 142367 AND o_orderdate = '1994-01-29' AND o_shippriority < 0 GROUP BY c_nationkey, n_name, o_clerk, r_name;
SELECT c_address, o_shippriority FROM orders JOIN customer ON c_custkey = o_custkey WHERE c_name > 'Customer#000014181' AND c_mktsegment <> 'FURNITURE' AND o_clerk = 'Clerk#000000072' AND o_totalprice <> 44719.4;
SELECT ps_suppkey, p_retailprice, s_nationkey FROM part JOIN partsupp ON ps_partkey = p_partkey JOIN supplier ON s_suppkey = ps_suppkey WHERE s_acctbal < 824.94 AND ps_partkey <= 27306 AND s_comment >= 'ind carefully above the escapades. slyly even requests' AND ps_supplycost < 996.79;
SELECT p_name, ps_availqty FROM partsupp, part WHERE p_container <= 'LG CASE' AND p_brand > 'Brand#32' AND ps_comment = 'e carefully pending foxes sleep furiously among the slyly special requests. blithely final req' AND p_type > 'MEDIUM BRUSHED NICKEL' ORDER BY ps_availqty ASC, p_name ASC;
SELECT n_nationkey, s_acctbal, c_custkey FROM supplier, nation, customer WHERE c_nationkey > 1 AND s_suppkey = 2138 AND s_comment <= 'fluffily furiously pending accoun' AND s_name <> 'Supplier#000004985' AND c_name >= 'Customer#000140806' ORDER BY s_acctbal ASC;
SELECT ps_suppkey, l_tax FROM partsupp, lineitem WHERE l_orderkey <> 3225285 AND l_commitdate = '1997-12-11' AND ps_supplycost <> 757.42 AND l_discount = 0.08 AND ps_availqty >= 9820;
