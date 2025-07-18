CLASS zcl_test_humaneval_149 DEFINITION
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

CLASS zcl_test_humaneval_149 IMPLEMENTATION.

  METHOD test_case_1.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    input = VALUE #( ( `aa` ) ( `a` ) ( `aaa` ) ).
    expected = VALUE #( ( `aa` ) ).
    actual = z_humaneval_149=>sorted_list_sum( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    input = VALUE #( ( `school` ) ( `AI` ) ( `asdf` ) ( `b` ) ).
    expected = VALUE #( ( `AI` ) ( `asdf` ) ( `school` ) ).
    actual = z_humaneval_149=>sorted_list_sum( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    input = VALUE #( ( `d` ) ( `b` ) ( `c` ) ( `a` ) ).
    expected = VALUE #( ).
    actual = z_humaneval_149=>sorted_list_sum( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    input = VALUE #( ( `d` ) ( `dcba` ) ( `abcd` ) ( `a` ) ).
    expected = VALUE #( ( `abcd` ) ( `dcba` ) ).
    actual = z_humaneval_149=>sorted_list_sum( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    input = VALUE #( ( `AI` ) ( `ai` ) ( `au` ) ).
    expected = VALUE #( ( `AI` ) ( `ai` ) ( `au` ) ).
    actual = z_humaneval_149=>sorted_list_sum( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    input = VALUE #( ( `a` ) ( `b` ) ( `b` ) ( `c` ) ( `c` ) ( `a` ) ).
    expected = VALUE #( ).
    actual = z_humaneval_149=>sorted_list_sum( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: input    TYPE string_table,
          expected TYPE string_table,
          actual   TYPE string_table.

    input = VALUE #( ( `aaaa` ) ( `bbbb` ) ( `dd` ) ( `cc` ) ).
    expected = VALUE #( ( `cc` ) ( `dd` ) ( `aaaa` ) ( `bbbb` ) ).
    actual = z_humaneval_149=>sorted_list_sum( input ).

    cl_abap_unit_assert=>assert_equals(
      act = actual
      exp = expected
    ).
  ENDMETHOD.

ENDCLASS.