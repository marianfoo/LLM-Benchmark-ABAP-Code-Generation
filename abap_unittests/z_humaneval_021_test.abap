CLASS lhc_z_humaneval_021 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_number,
             value TYPE f,
           END OF ty_number.
    TYPES: tt_numbers TYPE STANDARD TABLE OF ty_number.
    TYPES: tt_result TYPE STANDARD TABLE OF f WITH NON-UNIQUE DEFAULT KEY.
    METHODS test_case_1 FOR TESTING.
    METHODS test_case_2 FOR TESTING.
    METHODS test_case_3 FOR TESTING.
    METHODS test_case_4 FOR TESTING.
    METHODS test_case_5 FOR TESTING.
    METHODS assert_float_equals
      IMPORTING
        VALUE(iv_actual)   TYPE f
        VALUE(iv_expected) TYPE f
        VALUE(iv_message)  TYPE string.
ENDCLASS.

CLASS lhc_z_humaneval_021 IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lt_numbers  TYPE tt_numbers,
          lt_result   TYPE tt_result,
          lt_expected TYPE STANDARD TABLE OF f.

    lt_numbers = VALUE #( ( value = 2 ) ( value = '49.9' ) ).
    lt_result = z_humaneval_021=>rescale_to_unit( numbers = lt_numbers ).
    lt_expected = VALUE #( ( CONV f( 0 ) ) ( CONV f( 1 ) ) ).

    LOOP AT lt_result INTO DATA(result_entry).
      READ TABLE lt_expected INDEX sy-tabix INTO DATA(expected_value).
      me->assert_float_equals(
        iv_actual = result_entry
        iv_expected = expected_value
        iv_message = |Test case 1 failed at index { sy-tabix }| ).
    ENDLOOP.
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_numbers  TYPE tt_numbers,
          lt_result   TYPE tt_result,
          lt_expected TYPE STANDARD TABLE OF f.

    lt_numbers = VALUE #( ( value = 100 ) ( value = '49.9' ) ).
    lt_result = z_humaneval_021=>rescale_to_unit( numbers = lt_numbers ).
    lt_expected = VALUE #( ( CONV f( 1 ) ) ( CONV f( 0 ) ) ).

    LOOP AT lt_result INTO DATA(result_entry).
      READ TABLE lt_expected INDEX sy-tabix INTO DATA(expected_value).
      me->assert_float_equals(
        iv_actual = result_entry
        iv_expected = expected_value
        iv_message = |Test case 2 failed at index { sy-tabix }| ).
    ENDLOOP.
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_numbers  TYPE tt_numbers,
          lt_result   TYPE tt_result,
          lt_expected TYPE STANDARD TABLE OF f.

    lt_numbers = VALUE #( ( value = 1 ) ( value = 2 ) ( value = 3 ) ( value = 4 ) ( value = 5 ) ).
    lt_result = z_humaneval_021=>rescale_to_unit( numbers = lt_numbers ).
    lt_expected = VALUE #( ( CONV f( 0 ) ) ( CONV f( '0.25' ) ) ( CONV f( '0.5' ) ) ( CONV f( '0.75' ) ) ( CONV f( 1 ) ) ).

    LOOP AT lt_result INTO DATA(result_entry).
      READ TABLE lt_expected INDEX sy-tabix INTO DATA(expected_value).
      me->assert_float_equals(
        iv_actual = result_entry
        iv_expected = expected_value
        iv_message = |Test case 3 failed at index { sy-tabix }| ).
    ENDLOOP.
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lt_numbers  TYPE tt_numbers,
          lt_result   TYPE tt_result,
          lt_expected TYPE STANDARD TABLE OF f.

    lt_numbers = VALUE #( ( value = 2 ) ( value = 1 ) ( value = 5 ) ( value = 3 ) ( value = 4 ) ).
    lt_result = z_humaneval_021=>rescale_to_unit( numbers = lt_numbers ).
    lt_expected = VALUE #( ( CONV f( '0.25' ) ) ( CONV f( 0 ) ) ( CONV f( 1 ) ) ( CONV f( '0.5' ) ) ( CONV f( '0.75' ) ) ).

    LOOP AT lt_result INTO DATA(result_entry).
      READ TABLE lt_expected INDEX sy-tabix INTO DATA(expected_value).
      me->assert_float_equals(
        iv_actual = result_entry
        iv_expected = expected_value
        iv_message = |Test case 4 failed at index { sy-tabix }| ).
    ENDLOOP.
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lt_numbers  TYPE tt_numbers,
          lt_result   TYPE tt_result,
          lt_expected TYPE STANDARD TABLE OF f.

    lt_numbers = VALUE #( ( value = 12 ) ( value = 11 ) ( value = 15 ) ( value = 13 ) ( value = 14 ) ).
    lt_result = z_humaneval_021=>rescale_to_unit( numbers = lt_numbers ).
    lt_expected = VALUE #( ( CONV f( '0.25' ) ) ( CONV f( 0 ) ) ( CONV f( 1 ) ) ( CONV f( '0.5' ) ) ( CONV f( '0.75' ) ) ).

    LOOP AT lt_result INTO DATA(result_entry).
      READ TABLE lt_expected INDEX sy-tabix INTO DATA(expected_value).
      me->assert_float_equals(
        iv_actual = result_entry
        iv_expected = expected_value
        iv_message = |Test case 5 failed at index { sy-tabix }| ).
    ENDLOOP.
  ENDMETHOD.

  METHOD assert_float_equals.
    DATA: lv_delta TYPE f VALUE '0.00001'.

    IF abs( iv_actual - iv_expected ) > lv_delta.
      cl_abap_unit_assert=>fail(
        msg = |{ iv_message }: Expected { iv_expected }, but got { iv_actual }| ).
    ENDIF.
  ENDMETHOD.

ENDCLASS.