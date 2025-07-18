CLASS ltcl_test_has_close_elements DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    TYPES: BEGIN OF ty_float_table,
             value TYPE f,
           END OF ty_float_table.
    TYPES: tt_float_table TYPE TABLE OF ty_float_table.

    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_has_close_elements IMPLEMENTATION.
  METHOD test_case_1.
    DATA: numbers TYPE tt_float_table,
          result  TYPE abap_bool.

    numbers = VALUE #( ( value = '1.0' ) ( value = '2.0' ) ( value = '3.9' ) ( value = '4.0' ) ( value = '5.0' ) ( value = '2.2' ) ).
    result = z_humaneval_000=>has_close_elements( numbers = numbers threshold = '0.3' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: numbers TYPE tt_float_table,
          result  TYPE abap_bool.

    numbers = VALUE #( ( value = '1.0' ) ( value = '2.0' ) ( value = '3.9' ) ( value = '4.0' ) ( value = '5.0' ) ( value = '2.2' ) ).
    result = z_humaneval_000=>has_close_elements( numbers = numbers threshold = '0.05' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: numbers TYPE tt_float_table,
          result  TYPE abap_bool.

    numbers = VALUE #( ( value = '1.0' ) ( value = '2.0' ) ( value = '5.9' ) ( value = '4.0' ) ( value = '5.0' ) ).
    result = z_humaneval_000=>has_close_elements( numbers = numbers threshold = '0.95' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: numbers TYPE tt_float_table,
          result  TYPE abap_bool.

    numbers = VALUE #( ( value = '1.0' ) ( value = '2.0' ) ( value = '5.9' ) ( value = '4.0' ) ( value = '5.0' ) ).
    result = z_humaneval_000=>has_close_elements( numbers = numbers threshold = '0.8' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: numbers TYPE tt_float_table,
          result  TYPE abap_bool.

    numbers = VALUE #( ( value = '1.0' ) ( value = '2.0' ) ( value = '3.0' ) ( value = '4.0' ) ( value = '5.0' ) ( value = '2.0' ) ).
    result = z_humaneval_000=>has_close_elements( numbers = numbers threshold = '0.1' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: numbers TYPE tt_float_table,
          result  TYPE abap_bool.

    numbers = VALUE #( ( value = '1.1' ) ( value = '2.2' ) ( value = '3.1' ) ( value = '4.1' ) ( value = '5.1' ) ).
    result = z_humaneval_000=>has_close_elements( numbers = numbers threshold = '1.0' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: numbers TYPE tt_float_table,
          result  TYPE abap_bool.

    numbers = VALUE #( ( value = '1.1' ) ( value = '2.2' ) ( value = '3.1' ) ( value = '4.1' ) ( value = '5.1' ) ).
    result = z_humaneval_000=>has_close_elements( numbers = numbers threshold = '0.5' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = abap_false ).
  ENDMETHOD.
ENDCLASS.