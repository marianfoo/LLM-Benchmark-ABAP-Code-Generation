CLASS ltcl_test_pluck DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_pluck IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 4 ) ( 2 ) ( 3 ) ).
    expected = VALUE #( ( 2 ) ( 1 ) ).
    actual = z_humaneval_068=>pluck( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Error' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).
    expected = VALUE #( ( 2 ) ( 1 ) ).
    actual = z_humaneval_068=>pluck( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Error' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ).
    expected = VALUE #( ).
    actual = z_humaneval_068=>pluck( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Error' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 0 ) ( 3 ) ( 0 ) ( 4 ) ( 2 ) ).
    expected = VALUE #( ( 0 ) ( 1 ) ).
    actual = z_humaneval_068=>pluck( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Error' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ( 0 ) ( 5 ) ( 3 ) ).
    expected = VALUE #( ( 0 ) ( 3 ) ).
    actual = z_humaneval_068=>pluck( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Error' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 5 ) ( 4 ) ( 8 ) ( 4 ) ( 8 ) ).
    expected = VALUE #( ( 4 ) ( 1 ) ).
    actual = z_humaneval_068=>pluck( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Error' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 7 ) ( 6 ) ( 7 ) ( 1 ) ).
    expected = VALUE #( ( 6 ) ( 1 ) ).
    actual = z_humaneval_068=>pluck( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Error' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: input    TYPE int4_table,
          expected TYPE int4_table,
          actual   TYPE int4_table.

    input = VALUE #( ( 7 ) ( 9 ) ( 7 ) ( 1 ) ).
    expected = VALUE #( ).
    actual = z_humaneval_068=>pluck( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
      msg = 'Error' ).
  ENDMETHOD.

ENDCLASS.