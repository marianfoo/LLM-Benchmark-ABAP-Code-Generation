CLASS ltcl_test_z_humaneval_123 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_14 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_12 FOR TESTING,
      test_case_1 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_z_humaneval_123 IMPLEMENTATION.
  METHOD test_case_14.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 5 ) ( 7 ) ( 11 ) ( 13 ) ( 17 ) ).
    result = z_humaneval_123=>get_odd_collatz( 14 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 5 ) ).
    result = z_humaneval_123=>get_odd_collatz( 5 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_12.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    expected = VALUE #( ( 1 ) ( 3 ) ( 5 ) ).
    result = z_humaneval_123=>get_odd_collatz( 12 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_1.
    DATA: result   TYPE int4_table,
          expected TYPE int4_table.

    expected = VALUE #( ( 1 ) ).
    result = z_humaneval_123=>get_odd_collatz( 1 ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
    ).
  ENDMETHOD.
ENDCLASS.