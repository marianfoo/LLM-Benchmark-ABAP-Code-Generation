CLASS ltc_sort_third_test DEFINITION
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
      test_case_7 FOR TESTING.

ENDCLASS.

CLASS ltc_sort_third_test IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).
    expected = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).

    actual = z_humaneval_033=>sort_third( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test case 1 failed'
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 3 ) ( -5 ) ( 2 ) ( -3 ) ( 3 ) ( 9 ) ( 0 ) ( 123 ) ( 1 ) ( -10 ) ).
    expected = VALUE #( ( 1 ) ( 3 ) ( -5 ) ( 2 ) ( -3 ) ( 3 ) ( 5 ) ( 0 ) ( 123 ) ( 9 ) ( -10 ) ).

    actual = z_humaneval_033=>sort_third( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test case 2 failed'
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 8 ) ( -12 ) ( 4 ) ( 23 ) ( 2 ) ( 3 ) ( 11 ) ( 12 ) ( -10 ) ).
    expected = VALUE #( ( -10 ) ( 8 ) ( -12 ) ( 3 ) ( 23 ) ( 2 ) ( 4 ) ( 11 ) ( 12 ) ( 5 ) ).

    actual = z_humaneval_033=>sort_third( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test case 3 failed'
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 6 ) ( 3 ) ( 4 ) ( 8 ) ( 9 ) ( 2 ) ).
    expected = VALUE #( ( 2 ) ( 6 ) ( 3 ) ( 4 ) ( 8 ) ( 9 ) ( 5 ) ).

    actual = z_humaneval_033=>sort_third( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test case 4 failed'
    ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 8 ) ( 3 ) ( 4 ) ( 6 ) ( 9 ) ( 2 ) ).
    expected = VALUE #( ( 2 ) ( 8 ) ( 3 ) ( 4 ) ( 6 ) ( 9 ) ( 5 ) ).

    actual = z_humaneval_033=>sort_third( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test case 5 failed'
    ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 6 ) ( 9 ) ( 4 ) ( 8 ) ( 3 ) ( 2 ) ).
    expected = VALUE #( ( 2 ) ( 6 ) ( 9 ) ( 4 ) ( 8 ) ( 3 ) ( 5 ) ).

    actual = z_humaneval_033=>sort_third( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test case 6 failed'
    ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 6 ) ( 3 ) ( 4 ) ( 8 ) ( 9 ) ( 2 ) ( 1 ) ).
    expected = VALUE #( ( 2 ) ( 6 ) ( 3 ) ( 4 ) ( 8 ) ( 9 ) ( 5 ) ( 1 ) ).

    actual = z_humaneval_033=>sort_third( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Test case 7 failed'
    ).
  ENDMETHOD.

ENDCLASS.