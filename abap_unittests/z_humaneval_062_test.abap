CLASS ltcl_test_derivative DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_derivative IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lv_input    TYPE int4_table,
          lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_input = VALUE #( ( 3 ) ( 1 ) ( 2 ) ( 4 ) ( 5 ) ).
    lv_expected = VALUE #( ( 1 ) ( 4 ) ( 12 ) ( 20 ) ).
    lv_actual = z_humaneval_062=>derivative( lv_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lv_input    TYPE int4_table,
          lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_input = VALUE #( ( 1 ) ( 2 ) ( 3 ) ).
    lv_expected = VALUE #( ( 2 ) ( 6 ) ).
    lv_actual = z_humaneval_062=>derivative( lv_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lv_input    TYPE int4_table,
          lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_input = VALUE #( ( 3 ) ( 2 ) ( 1 ) ).
    lv_expected = VALUE #( ( 2 ) ( 2 ) ).
    lv_actual = z_humaneval_062=>derivative( lv_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lv_input    TYPE int4_table,
          lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_input = VALUE #( ( 3 ) ( 2 ) ( 1 ) ( 0 ) ( 4 ) ).
    lv_expected = VALUE #( ( 2 ) ( 2 ) ( 0 ) ( 16 ) ).
    lv_actual = z_humaneval_062=>derivative( lv_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lv_input    TYPE int4_table,
          lv_expected TYPE int4_table,
          lv_actual   TYPE int4_table.

    lv_input = VALUE #( ( 1 ) ).
    lv_expected = VALUE #( ).
    lv_actual = z_humaneval_062=>derivative( lv_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_actual
      exp = lv_expected ).
  ENDMETHOD.

ENDCLASS.