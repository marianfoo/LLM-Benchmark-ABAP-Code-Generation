CLASS ltcl_test_sort_even DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_sort_even IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).
    expected = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).

    actual = z_humaneval_037=>sort_even( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 3 ) ( -5 ) ( 2 ) ( -3 ) ( 3 ) ( 9 ) ( 0 ) ( 123 ) ( 1 ) ( -10 ) ).
    expected = VALUE #( ( -10 ) ( 3 ) ( -5 ) ( 2 ) ( -3 ) ( 3 ) ( 5 ) ( 0 ) ( 9 ) ( 1 ) ( 123 ) ).

    actual = z_humaneval_037=>sort_even( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 8 ) ( -12 ) ( 4 ) ( 23 ) ( 2 ) ( 3 ) ( 11 ) ( 12 ) ( -10 ) ).
    expected = VALUE #( ( -12 ) ( 8 ) ( 3 ) ( 4 ) ( 5 ) ( 2 ) ( 12 ) ( 11 ) ( 23 ) ( -10 ) ).

    actual = z_humaneval_037=>sort_even( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

ENDCLASS.