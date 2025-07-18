CLASS ltcl_test_unique_digits DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_int_table,
             value TYPE i,
           END OF ty_int_table,
           tt_int_table TYPE STANDARD TABLE OF ty_int_table WITH NON-UNIQUE DEFAULT KEY.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_unique_digits IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lt_input    TYPE tt_int_table,
          lt_expected TYPE tt_int_table,
          lt_actual   TYPE tt_int_table.

    lt_input = VALUE #( ( value = 15 ) ( value = 33 ) ( value = 1422 ) ( value = 1 ) ).
    lt_expected = VALUE #( ( value = 1 ) ( value = 15 ) ( value = 33 ) ).

    lt_actual = z_humaneval_104=>unique_digits( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_input    TYPE tt_int_table,
          lt_expected TYPE tt_int_table,
          lt_actual   TYPE tt_int_table.

    lt_input = VALUE #( ( value = 152 ) ( value = 323 ) ( value = 1422 ) ( value = 10 ) ).
    CLEAR lt_expected.

    lt_actual = z_humaneval_104=>unique_digits( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_input    TYPE tt_int_table,
          lt_expected TYPE tt_int_table,
          lt_actual   TYPE tt_int_table.

    lt_input = VALUE #( ( value = 12345 ) ( value = 2033 ) ( value = 111 ) ( value = 151 ) ).
    lt_expected = VALUE #( ( value = 111 ) ( value = 151 ) ).

    lt_actual = z_humaneval_104=>unique_digits( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lt_input    TYPE tt_int_table,
          lt_expected TYPE tt_int_table,
          lt_actual   TYPE tt_int_table.

    lt_input = VALUE #( ( value = 135 ) ( value = 103 ) ( value = 31 ) ).
    lt_expected = VALUE #( ( value = 31 ) ( value = 135 ) ).

    lt_actual = z_humaneval_104=>unique_digits( lt_input ).

    cl_abap_unit_assert=>assert_equals(
      act = lt_actual
      exp = lt_expected ).
  ENDMETHOD.

ENDCLASS.