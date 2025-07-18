CLASS z_humaneval_145_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS test_case_1 FOR TESTING.
    METHODS test_case_2 FOR TESTING.
    METHODS test_case_3 FOR TESTING.
    METHODS test_case_4 FOR TESTING.
    METHODS test_case_5 FOR TESTING.
    METHODS test_case_6 FOR TESTING.

ENDCLASS.

CLASS z_humaneval_145_test IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( 11 ) ( -1 ) ( -11 ) ( -12 ) ).
    expected = VALUE #( ( -1 ) ( -11 ) ( 1 ) ( -12 ) ( 11 ) ).
    actual = z_humaneval_145=>order_by_points( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1234 ) ( 423 ) ( 463 ) ( 145 ) ( 2 ) ( 423 ) ( 423 ) ( 53 ) ( 6 ) ( 37 ) ( 3457 ) ( 3 ) ( 56 ) ( 0 ) ( 46 ) ).
    expected = VALUE #( ( 0 ) ( 2 ) ( 3 ) ( 6 ) ( 53 ) ( 423 ) ( 423 ) ( 423 ) ( 1234 ) ( 145 ) ( 37 ) ( 46 ) ( 56 ) ( 463 ) ( 3457 ) ).
    actual = z_humaneval_145=>order_by_points( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ).
    expected = VALUE #( ).
    actual = z_humaneval_145=>order_by_points( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( -11 ) ( -32 ) ( 43 ) ( 54 ) ( -98 ) ( 2 ) ( -3 ) ).
    expected = VALUE #( ( -3 ) ( -32 ) ( -98 ) ( -11 ) ( 1 ) ( 2 ) ( 43 ) ( 54 ) ).
    actual = z_humaneval_145=>order_by_points( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ( 6 ) ( 7 ) ( 8 ) ( 9 ) ( 10 ) ( 11 ) ).
    expected = VALUE #( ( 1 ) ( 10 ) ( 2 ) ( 11 ) ( 3 ) ( 4 ) ( 5 ) ( 6 ) ( 7 ) ( 8 ) ( 9 ) ).
    actual = z_humaneval_145=>order_by_points( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 0 ) ( 6 ) ( 6 ) ( -76 ) ( -21 ) ( 23 ) ( 4 ) ).
    expected = VALUE #( ( -76 ) ( -21 ) ( 0 ) ( 4 ) ( 23 ) ( 6 ) ( 6 ) ).
    actual = z_humaneval_145=>order_by_points( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

ENDCLASS.