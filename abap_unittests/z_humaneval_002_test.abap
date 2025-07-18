CLASS ltcl_truncate_number DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_truncate_3_5 FOR TESTING,
      test_truncate_1_33 FOR TESTING,
      test_truncate_123_456 FOR TESTING.

ENDCLASS.

CLASS ltcl_truncate_number IMPLEMENTATION.

  METHOD test_truncate_3_5.
    DATA: lv_input    TYPE f VALUE '3.5',
          lv_expected TYPE f VALUE '0.5',
          lv_result   TYPE f,
          lv_delta    TYPE f VALUE '0.000001'.

    lv_result = z_humaneval_002=>truncate_number( lv_input ).

    cl_abap_unit_assert=>assert_true(
      act = xsdbool( abs( lv_result - lv_expected ) LE lv_delta )
      msg = |Test case 3.5 failed: expected { lv_expected }, got { lv_result }| ).
  ENDMETHOD.

  METHOD test_truncate_1_33.
    DATA: lv_input    TYPE f VALUE '1.33',
          lv_expected TYPE f VALUE '0.33',
          lv_result   TYPE f,
          lv_delta    TYPE f VALUE '0.000001'.

    lv_result = z_humaneval_002=>truncate_number( lv_input ).

    cl_abap_unit_assert=>assert_true(
      act = xsdbool( abs( lv_result - lv_expected ) LE lv_delta )
      msg = |Test case 1.33 failed: expected { lv_expected }, got { lv_result }| ).
  ENDMETHOD.

  METHOD test_truncate_123_456.
    DATA: lv_input    TYPE f VALUE '123.456',
          lv_expected TYPE f VALUE '0.456',
          lv_result   TYPE f,
          lv_delta    TYPE f VALUE '0.000001'.

    lv_result = z_humaneval_002=>truncate_number( lv_input ).

    cl_abap_unit_assert=>assert_true(
      act = xsdbool( abs( lv_result - lv_expected ) LE lv_delta )
      msg = |Test case 123.456 failed: expected { lv_expected }, got { lv_result }| ).
  ENDMETHOD.

ENDCLASS.