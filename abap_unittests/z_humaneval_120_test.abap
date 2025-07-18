CLASS ltcl_test_maximum DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_11 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_maximum IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( -3 ) ( -4 ) ( 5 ) ).
    expected = VALUE #( ( -4 ) ( -3 ) ( 5 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 3 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 4 ) ( -4 ) ( 4 ) ).
    expected = VALUE #( ( 4 ) ( 4 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 2 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( -3 ) ( 2 ) ( 1 ) ( 2 ) ( -1 ) ( -2 ) ( 1 ) ).
    expected = VALUE #( ( 2 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 1 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 123 ) ( -123 ) ( 20 ) ( 0 ) ( 1 ) ( 2 ) ( -3 ) ).
    expected = VALUE #( ( 2 ) ( 20 ) ( 123 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 3 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( -123 ) ( 20 ) ( 0 ) ( 1 ) ( 2 ) ( -3 ) ).
    expected = VALUE #( ( 0 ) ( 1 ) ( 2 ) ( 20 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 4 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 15 ) ( 0 ) ( 3 ) ( -13 ) ( -8 ) ( 0 ) ).
    expected = VALUE #( ( -13 ) ( -8 ) ( 0 ) ( 0 ) ( 3 ) ( 5 ) ( 15 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 7 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( -1 ) ( 0 ) ( 2 ) ( 5 ) ( 3 ) ( -10 ) ).
    expected = VALUE #( ( 3 ) ( 5 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 2 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( 0 ) ( 5 ) ( -7 ) ).
    expected = VALUE #( ( 5 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 1 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 4 ) ( -4 ) ).
    expected = VALUE #( ( -4 ) ( 4 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 2 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( -10 ) ( 10 ) ).
    expected = VALUE #( ( -10 ) ( 10 ) ).
    actual = z_humaneval_120=>maximum( arr = input k = 2 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( -23 ) ( 243 ) ( -400 ) ( 0 ) ).
    CLEAR expected.
    actual = z_humaneval_120=>maximum( arr = input k = 0 ).
    cl_abap_unit_assert=>assert_equals( act = actual exp = expected ).
  ENDMETHOD.

ENDCLASS.